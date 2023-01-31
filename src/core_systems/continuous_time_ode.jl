using OrdinaryDiffEq: Tsit5
using SciMLBase: ODEProblem, DEIntegrator, u_modified!, __init
export CoupledODEs, ContinuousDynamicalSystem

##################################################################################
# DiffEq options
##################################################################################
_get_solver(a) = haskey(a, :alg) ? a[:alg] : DEFAULT_SOLVER
const DEFAULT_SOLVER = Tsit5()
const DEFAULT_DIFFEQ_KWARGS = (abstol = 1e-6, reltol = 1e-6)
const DEFAULT_DIFFEQ = (alg = DEFAULT_SOLVER, DEFAULT_DIFFEQ_KWARGS...)

# Function from user `@xlxs4`, see
# https://github.com/JuliaDynamics/DynamicalSystemsBase.jl/pull/153
_delete(a::NamedTuple, s::Symbol) = NamedTuple{filter(≠(s), keys(a))}(a)
function _decompose_into_solver_and_remaining(diffeq)
    if haskey(diffeq, :alg)
        return (diffeq[:alg], _delete(diffeq, :alg))
    else
        return (DEFAULT_SOLVER, diffeq)
    end
end

##################################################################################
# Type
##################################################################################
"""
    CoupledODEs(f, u0 [, p]; diffeq, t0 = 0.0) <: DynamicalSystem

A deterministic continuous time dynamical system defined by a set of
coupled ordinary differential equations as follows:
```math
\\frac{d\\vec{u}}{dt} = \\vec{f}(\\vec{u}, p, t)
```
An alias for `CoupledODE` is `ContinuousDynamicalSystem`.

Optionally provide the parameter container `p` and initial time as keyword `t0`.

For construction instructions regarding `f, u0` see [`DynamicalSystem`](@ref).

## DifferentialEquations.jl keyword arguments and interfacing

The ODEs are evolved via the solvers of DifferentialEquations.jl.
When initializing a `CoupledODEs`, you can specify the solver that will integrate
`f` in time, along with any other integration options, using the `diffeq` keyword.
For example you could use `diffeq = (abstol = 1e-9, reltol = 1e-9)`.
If you want to specify a solver, do so by using the keyword `alg`, e.g.:
`diffeq = (alg = Tsit5(), maxiters = 100000)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers. The default `diffeq` is:

$(DynamicalSystemsBase.DEFAULT_DIFFEQ)

`diffeq` keywords can also include `callback` for [event handling
](http://docs.juliadiffeq.org/latest/features/callback_functions.html), however the
majority of downstream functions in DynamicalSystems.jl assume that `f` is differentiable.

The convenience constructor `CoupledODEs(prob::ODEProblem, diffeq)` is also available.

Dev note: `CoupledODEs` is a light wrapper of `ODEIntegrator` from DifferentialEquations.jl.
The integrator is available as the field `integ`, and the `ODEProblem` is `integ.sol.prob`.
"""
struct CoupledODEs{D, I, P, E} <: ContinuousTimeDynamicalSystem
    integ::I
    # initial parameter container is the only field we can't recover from `integ`
    p0::P
    diffeq::E
end

"""
    ContinuousDynamicalSystem

An alias to [`CoupledODEs`](@ref).
This was the name these systems had before DynamicalSystems.jl v3.0.
"""
const ContinuousDynamicalSystem = CoupledODEs

function CoupledODEs(f, u0, p = SciMLBase.NullParameters(); t0 = 0, diffeq = DEFAULT_DIFFEQ)
    IIP = isinplace(f, 4) # from SciMLBase
    s = correct_state_type(Val{IIP}(), u0)
    # Initialize integrator
    T = eltype(s)
    prob = ODEProblem{IIP}(f, s, (T(t0), T(Inf)), p)
    return CoupledODEs(prob, diffeq)
end
function CoupledODEs(prob::ODEProblem, diffeq = DEFAULT_DIFFEQ)
    D = length(prob.u0)
    P = typeof(prob.p)
    solver, remaining = _decompose_into_solver_and_remaining(diffeq)
    integ = __init(prob, solver; remaining...,
        # Integrators are used exclusively iteratively. There is no reason to save anything.
        save_start = false, save_end = false, save_everystep = false
    )
    return CoupledODEs{D, typeof(integ), P}(integ, deepcopy(prob.p), diffeq)
end

##################################################################################
# Extend interface and extend for `DEIntegrator`
##################################################################################
StateSpaceSets.dimension(::CoupledODEs{D}) where {D} = D

for f in (:current_state, :initial_state, :current_parameters, :dynamic_rule,
    :current_time, :initial_time, :set_state!, :(SciMLBase.step!))
    @eval $(f)(ds::ContinuousTimeDynamicalSystem, args...) = $(f)(ds.integ, args...)
end

SciMLBase.isinplace(ds::ContinuousTimeDynamicalSystem) = isinplace(ds.integ.f)

function SciMLBase.reinit!(ds::ContinuousTimeDynamicalSystem, u = initial_state(ds);
        p0 = current_parameters(ds), t0 = initial_time(ds)
    )
    isnothing(u) && return
    set_parameters!(ds, p0)
    reinit!(ds.integ, u; reset_dt = true, t0)
end

# `DEIntegrator` stuff
dynamic_rule(integ::DEIntegrator) = integ.f.f
current_parameters(integ::DEIntegrator) = integ.p
initial_state(integ::DEIntegrator) = integ.sol.prob.u0
current_state(integ::DEIntegrator) = integ.u
current_time(integ::DEIntegrator) = integ.t
initial_time(integ::DEIntegrator) = integ.sol.prob.tspan[1]

function set_state!(integ::DEIntegrator, u)
    integ.u = u
    u_modified!(integ, true)
    return
end
