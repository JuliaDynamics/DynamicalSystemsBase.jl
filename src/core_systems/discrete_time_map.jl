# Implementation of concrete `DeterministicIteratedMap`
export DeterministicIteratedMap, DiscreteDynamicalSystem

###########################################################################################
# Type
###########################################################################################
"""
    DeterministicIteratedMap <: DynamicalSystem
    DeterministicIteratedMap(f, u0, p = nothing; t0 = 0)

A deterministic discrete time dynamical system defined by an iterated map as follows:
```math
\\vec{u}_{n+1} = \\vec{f}(\\vec{u}_n, p, n)
```
An alias for `DeterministicIteratedMap` is `DiscreteDynamicalSystem`.

Optionally configure the parameter container `p` and initial time `t0`.

For construction instructions regarding `f, u0` see [`DynamicalSystem`](@ref).
"""
mutable struct DeterministicIteratedMap{IIP, S, D, F, P} <: DiscreteTimeDynamicalSystem
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
This was the name these systems had before DynamicalSystems.jl v3.0.
"""
const DiscreteDynamicalSystem = DeterministicIteratedMap

function DeterministicIteratedMap(f, u0, p = nothing; t0::Integer = 0)
    IIP = isinplace(f, 4) # from SciMLBase
    s = correct_state(Val{IIP}(), u0)
    S = typeof(s)
    D = length(s)
    P = typeof(p)
    F = typeof(f)
    dummy = recursivecopy(s)
    # Before initialization do a sanity check that the rule and state match
    if !IIP
        out = f(s, p, t0)
        if typeof(out) != S
            error("Dynamic rule does not output correct state! "*
            "Got $(typeof(out)) instead of $(S).")
        end
    else
        out = f(dummy, s, p, t0)
        if !isnothing(out)
            error("Dynamic rule does not return `nothing`!")
        end
    end

    return DeterministicIteratedMap{IIP, S, D, F, P}(
        f, recursivecopy(s), recursivecopy(s), dummy, t0, t0, p, deepcopy(p)
    )
end

# Extend the interface components that aren't done by default:
SciMLBase.isinplace(::DIM{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::DIM{IIP, S, D}) where {IIP, S, D} = D
isdeterministic(::DIM) = true

function set_state!(ds::DeterministicIteratedMap{IIP}, u) where {IIP}
    ds.u = recursivecopy(u)
    ds.dummy = recursivecopy(u)
    return
end

###########################################################################################
# step!
###########################################################################################
# IIP version
function SciMLBase.step!(ds::DeterministicIteratedMap{true})
    # array swap
    ds.dummy, ds.u = ds.u, ds.dummy
    ds.f(ds.u, ds.dummy, ds.p, ds.t)
    ds.t += 1
    return ds
end
function SciMLBase.step!(ds::DeterministicIteratedMap{true}, N, stop_at_tdt = true)
    for _ in 1:N
        ds.dummy, ds.u = ds.u, ds.dummy
        ds.f(ds.u, ds.dummy, ds.p, ds.t)
        ds.t += 1
    end
    return ds
end

# OOP version
function SciMLBase.step!(ds::DeterministicIteratedMap{false})
    ds.u = ds.f(ds.u, ds.p, ds.t)
    ds.t +=1
    return ds
end
function SciMLBase.step!(ds::DeterministicIteratedMap{false}, N, stop_at_tdt = true)
    for _ in 1:N
        ds.u = ds.f(ds.u, ds.p, ds.t)
        ds.t += 1
    end
    return ds
end

###########################################################################################
# Alterations
###########################################################################################
function reinit!(ds::DIM, u = initial_state(ds);
        p = current_parameters(ds), t0 = initial_time(ds)
    )
    isnothing(u) && return
    set_state!(ds, u)
    ds.t = t0
    set_parameters!(ds, p)
    return ds
end
