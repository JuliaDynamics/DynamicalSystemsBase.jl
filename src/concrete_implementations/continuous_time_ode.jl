using SimpleDiffEq: SimpleATsit5
using SciMLBase: init
export CoupledODEs, ContinuousDynamicalSystem

##################################################################################
# DiffEq options
##################################################################################
_get_solver(a) = haskey(a, :alg) ? a[:alg] : DEFAULT_SOLVER
const DEFAULT_SOLVER = SimpleATsit5()
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
    CoupledODEs(f, u0, p = nothing; diffeq, t0 = 0.0) <: DynamicalSystem

A deterministic continuous time dynamical system defined by a set of
coupled ordinary differential equations as follows:
```math
\\frac{d\\vec{u}}{dt} = \\vec{f}(\\vec{u}, p, t)
```
An alias for `CoupledODE` is `ContinuousDynamicalSystem`.

Optionally configure the parameter container `p` and initial time as keyword `t0`.

For construction instructions regarding `f, u0` see [`DynamicalSystem`](@ref).

## DifferentialEquations.jl keyword arguments and interfacing

Continuous dynamical systems are evolved via the solvers of DifferentialEquations.jl.
When initializing a `CoupledODEs`, you can specify the solver that will integrate
`f` in time, along with any other integration options, using the `diffeq` keyword.
For example you could use `diffeq = (abstol = 1e-9, reltol = 1e-9)`.
If you want to specify a solver, do so by using the keyword `alg`, e.g.:
`diffeq = (alg = Tsit5(), maxiters = 100000)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers. The default `diffeq` is:

$(DynamicalSystemsBase.DEFAULT_DIFFEQ)

The default solver `SimpleATsit5` for simplicity and small load times.
It is strongly recommended to use a more featurefull and performant solver like
`Tsit5` or `Vern9` from OrdinaryDiffEq.jl.

`diffeq` keywords can also include `callback` for [event handling
](http://docs.juliadiffeq.org/latest/features/callback_functions.html), however the
majority of downstream functions in DynamicalSystems.jl assume that `f` is differentiable.

Dev note: `CoupledODEs` is a light wrapper of `ODEIntegrator` from DifferentialEquations.jl.
The integrator is available as the field `integ`, and the `ODEProblem` is `integ.sol.prob`.
"""
struct CoupledODEs{IIP, D, I, P} <: ContinuousTimeDynamicalSystem
    integ::I
    # initial parameter container is the only field we can't recover from `integ`
    p0::P
end

function CoupledODEs(f, u0, p = nothing; t0 = 0, diffeq = DEFAULT_DIFFEQ)
    IIP = isinplace(f, 4) # from SciMLBase
    s = correct_state_type(Val{IIP}(), u0)
    S = typeof(s)
    D = length(s)
    P = typeof(p)
    # Initialize integrator
    T = eltype(s)
    prob = ODEProblem{IIP}(f, s, (T(t0), T(Inf)), p)
    solver, remaining = _decompose_into_solver_and_remaining(diffeq)
    integ = __init(prob, solver; remaining..., save_everystep = false)
    return CoupledODEs{IIP, D, I, P}(integ, deepcopy(p))
end
