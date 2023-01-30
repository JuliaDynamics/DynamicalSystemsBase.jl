# Implementation of concrete `DeterministicIteratedMap`
export DeterministicIteratedMap, DiscreteDynamicalSystem

##################################################################################
# Type
##################################################################################
"""
    DeterministicIteratedMap(f, u0, p = nothing, t0 = 0) <: DynamicalSystem

A deterministic discrete time dynamical system defined by an iterated map as follows:
```math
\\vec{u}_{n+1} = \\vec{f}(\\vec{u}_n, p, n)
```
An alias for `DeterministicIteratedMap` is `DiscreteDynamicalSystem`.

Optionally configure the parameter container `p` and initial time `t0`.

For construction instructions regarding `f, u0` see [`DynamicalSystem`](@ref).
"""
mutable struct DeterministicIteratedMap{IIP, S, D, F, P} <: DynamicalSystem
    f::F
    u::S
    const u0::S
    dummy::S # dummy, used only in the IIP version
    t::Int
    const t0::Int
    p::P
    const p0::P
end
const DIM = DeterministicIteratedMap # Shortcut

"""
    DiscreteDynamicalSystem

An alias to [`DeterministicIteratedMap`](@ref).
This was the name these systems had before DynamicalSystems.jl 3.0.
"""
const DiscreteDynamicalSystem = DeterministicIteratedMap

# TODO: Also allow the deprecated method

function DeterministicIteratedMap(f, u0, p = nothing, t0::Integer = 0)
    IIP = isinplace(f, 4) # from SciMLBase
    s = correct_state_type(Val{IIP}(), u0)
    S = typeof(s)
    D = length(s)
    P = typeof(p)
    F = typeof(f)
    return DeterministicIteratedMap{IIP, S, D, F, P}(
        f, u0, deepcopy(u0), deepcopy(u0), t0, t0, p, deepcopy(p)
    )
end

# Extend the interface components that aren't done by default:
SciMLBase.isinplace(::DIM{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::DIM{IIP, S, D}) where {IIP, S, D} = D
isdiscretetime(::DIM) = true
isdeterministic(::DIM) = true

function (ds::DeterministicIteratedMap)(t::Real)
    if t == current_time(ds)
        return current_state(ds)
    end
    throw(ArgumentError("Cannot interpolate or extrapolate `DeterministicIteratedMap`."))
end

function set_state!(ds::DeterministicIteratedMap, u)
    if isinplace(ds)
        ds.u .= u
        ds.dummy .= u
    else
        ds.u = u
        ds.dummy = u
    end
    return
end

##################################################################################
# step!
##################################################################################
# IIP version
function SciMLBase.step!(ds::DeterministicIteratedMap{true})
    # array swap
    ds.dummy, ds.u = ds.u, ds.dummy
    ds.f(ds.u, ds.dummy, ds.p, ds.t)
    ds.t += 1
    return
end
function SciMLBase.step!(ds::DeterministicIteratedMap{true}, N)
    for _ in 1:N
        ds.dummy, ds.u = ds.u, ds.dummy
        ds.f(ds.u, ds.dummy, ds.p, ds.t)
        ds.t += 1
    end
    return
end

# OOP version
function SciMLBase.step!(ds::DeterministicIteratedMap{false})
    ds.u = ds.f(ds.u, ds.p, ds.t)
    ds.t +=1
    return
end
function SciMLBase.step!(ds::DeterministicIteratedMap{false}, N)
    for _ in 1:N
        ds.u = ds.f(ds.u, ds.p, ds.t)
        ds.t += 1
    end
    return
end

SciMLBase.step!(ds::DIM, N, stop_at_tdt) = SciMLBase.step!(ds, N)

##################################################################################
# Alterations
##################################################################################
function reinit!(ds::DIM, u = initial_state(ds);
        p0 = current_parameters(ds), t0 = initial_time(ds)
    )
    set_state!(ds, u)
    ds.t = t0
    set_parameters!(ds, p0)
    ds.p = p0
    return
end