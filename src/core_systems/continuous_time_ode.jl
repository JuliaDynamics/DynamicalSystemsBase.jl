using OrdinaryDiffEq: Tsit5
using SciMLBase: ODEProblem, DEIntegrator, u_modified!, __init
export CoupledODEs, ContinuousDynamicalSystem

###########################################################################################
# DiffEq options
###########################################################################################
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

###########################################################################################
# Type
###########################################################################################
"""
    CoupledODEs <: ContinuousTimeDynamicalSystem
    CoupledODEs(f, u0 [, p]; diffeq, t0 = 0.0)

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
The convenience syntax `ODEProblem(ds::CoupledODEs, tspan = (t0, Inf))` is available.
"""
struct CoupledODEs{IIP, D, I, P} <: ContinuousTimeDynamicalSystem
    integ::I
    # things we can't recover from `integ`
    p0::P
    diffeq # isn't parameterized because it is only used for display
end

"""
    ContinuousDynamicalSystem

An alias to [`CoupledODEs`](@ref).
This was the name these systems had before DynamicalSystems.jl v3.0.
"""
const ContinuousDynamicalSystem = CoupledODEs

function CoupledODEs(f, u0, p = SciMLBase.NullParameters(); t0 = 0, diffeq = DEFAULT_DIFFEQ)
    IIP = isinplace(f, 4) # from SciMLBase
    s = correct_state(Val{IIP}(), u0)
    T = eltype(s)
    prob = ODEProblem{IIP}(f, s, (T(t0), T(Inf)), p)
    return CoupledODEs(prob, diffeq)
end
# Below `special_kwargs` is undocumented internal option for passing `internalnorm`
function CoupledODEs(prob::ODEProblem, diffeq = DEFAULT_DIFFEQ; special_kwargs...)
    IIP = isinplace(prob)
    D = length(prob.u0)
    P = typeof(prob.p)
    solver, remaining = _decompose_into_solver_and_remaining(diffeq)
    integ = __init(prob, solver; remaining..., special_kwargs...,
        # Integrators are used exclusively iteratively. There is no reason to save anything.
        save_start = false, save_end = false, save_everystep = false,
        # DynamicalSystems.jl operates on integrators and `step!` exclusively,
        # so there is no reason to limit the maximum time evolution
        maxiters = Inf,
    )
    return CoupledODEs{IIP, D, typeof(integ), P}(integ, deepcopy(prob.p), diffeq)
end

function SciMLBase.ODEProblem(ds::CoupledODEs{IIP}, tspan = (initial_time(ds), Inf)) where {IIP}
    ODEProblem{IIP}(dynamic_rule(ds), initial_state(ds), tspan, current_parameters(ds))
end

###########################################################################################
# Extend interface and extend for `DEIntegrator`
###########################################################################################
StateSpaceSets.dimension(::CoupledODEs{IIP, D}) where {IIP, D} = D

for f in (:current_state, :initial_state, :current_parameters, :dynamic_rule,
    :current_time, :initial_time, :set_state!, :(SciMLBase.isinplace))
    @eval $(f)(ds::ContinuousTimeDynamicalSystem, args...) = $(f)(ds.integ, args...)
end

SciMLBase.isinplace(::CoupledODEs{IIP}) where {IIP} = IIP
# so that `ds` is printed
SciMLBase.step!(ds::CoupledODEs, args...) = (step!(ds.integ, args...); ds)

function SciMLBase.reinit!(ds::ContinuousTimeDynamicalSystem, u = initial_state(ds);
        p = current_parameters(ds), t0 = initial_time(ds)
    )
    isnothing(u) && return
    set_parameters!(ds, p)
    reinit!(ds.integ, u; reset_dt = true, t0)
    return ds
end

# `DEIntegrator` stuff
dynamic_rule(integ::DEIntegrator) = integ.f.f
current_parameters(integ::DEIntegrator) = integ.p
initial_state(integ::DEIntegrator) = integ.sol.prob.u0
current_state(integ::DEIntegrator) = integ.u
current_time(integ::DEIntegrator) = integ.t
initial_time(integ::DEIntegrator) = integ.sol.prob.tspan[1]

function set_state!(integ::DEIntegrator, u)
    integ.u = recursivecopy(u)
    u_modified!(integ, true)
    return
end

# This is here to ensure that `u_modified!` is called
function set_parameter!(ds::CoupledODEs, args...)
    _set_parameter!(current_parameters(ds), args...)
    u_modified!(ds.integ, true)
    return
end

