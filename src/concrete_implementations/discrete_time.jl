# Implementation of concrete `DeterministicIterativeMap`

##################################################################################
# Type
##################################################################################
"""
    DeterministicIterativeMap(f, u0, p = nothing, t0 = 0) <: DynamicalSystem

A deterministic discrete time dynamical system defined by an iterative map as follows:
```math
\\vec{u}_{n+1} = \\vec{f}(\\vec{u}_n, p, n)
```
An alias for `DeterministicIterativeMap` is `DiscreteDynamicalSystem`.

Optionally configure the parameter container `p` and initial time `t0`.

For construction instructions regarding `f, u0` see [`DynamicalSystem`](@ref).
"""
mutable struct DeterministicIterativeMap{IIP, S, D, F, P}
    f::F      # integrator f
    u::S      # integrator state
    const u0::S
    dummy::S  # dummy, used only in the IIP version
    t::Int    # integrator "time" (counter)
    const t0::Int
    p::P      # parameter container
    const p0::P
end

"""
    DiscreteDynamicalSystem

An alias to [`DeterministicIterativeMap`](@ref).
This was the name these systems had before DynamicalSystems.jl 3.0.
"""
const DiscreteDynamicalSystem = DeterministicIterativeMap

function DeterministicIterativeMap(f, u0, p = nothing, t0 = 0)
    IIP = isinplace(f)
    s = safe_state_type(Val{IIP}(), u0)
    S = typeof(s)
    D = length(s)
    P = typeof(P)
    F = typeof(F)
    return DeterministicIterativeMap{IIP, S, D, F, P}(
        f, u0, deepcopy(u0), deepcopy(u0), t, t0, p, deepcopy(p)
    )
end

##################################################################################
# step!
##################################################################################
# IIP version
function step!(integ::DeterministicIterativeMap{true})
    # array swap
    integ.dummy, integ.u = integ.u, integ.dummy
    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end
function step!(integ::DeterministicIterativeMap{true}, N)
    for _ in 1:N
        integ.dummy, integ.u = integ.u, integ.dummy
        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
function step!(integ::DeterministicIterativeMap{false})
    integ.u = integ.f(integ.u, integ.p, integ.t)
    integ.t +=1
    return
end
function step!(integ::DeterministicIterativeMap{false}, N)
    for _ in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
end

step!(integ::DeterministicIterativeMap, N, stop_at_tdt) = step!(integ, N)