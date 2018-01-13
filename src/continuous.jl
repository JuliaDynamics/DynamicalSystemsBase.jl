using OrdinaryDiffEq, Requires, ForwardDiff, DiffEqCallbacks
import OrdinaryDiffEq.ODEProblem
import OrdinaryDiffEq.ODEIntegrator

export ContinuousDS, variational_integrator, ODEIntegrator, ODEProblem
export ContinuousDynamicalSystem, DEFAULT_DIFFEQ_KWARGS, get_sol

#######################################################################################
#                                     Constructors                                    #
#######################################################################################
"Abstract type representing continuous systems."
abstract type ContinuousDynamicalSystem <: DynamicalSystem end

"""
    ContinuousDS <: DynamicalSystem
`D`-dimensional continuous dynamical system.
## Fields
* `prob::ODEProblem` : The fundamental structure used to describe
  a continuous dynamical system and also used in the
  [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/index.html)
  ecosystem.
  Contains the system's state, the equations of motion and optionally other
  information like e.g. [callbacks](http://docs.juliadiffeq.org/latest/features/callback_functions.html#Event-Handling-and-Callback-Functions-1).
* `jacob!` (function) : The function that represents the Jacobian of the system,
  given in the format: `jacob!(t, u, J)` which means it is in-place, with the mutated
  argument being the last (`u` **must be** `Vector`).
* `J::Matrix{T}` : Jacobian matrix.

You can use `ds.prob.u0 .= newstate` to set a new state to the system.

## Creating a `ContinuousDS`
The equations of motion **must be** in the form `eom!(t, u, du)`,
which means that they are **in-place** with the mutated argument
`du` the last one. Both `u, du` **must be** `Vector`s. You can still use matrices
in your equations of motion though! Just change `function eom(t, u, du)` to
```julia
function eom(t, u, du)
    um = reshape(u, a, b); dum = reshape(du, a, b)
    # equations of motion with matrix shape of a×b
```
and you will be able to express the equations with matrix notation.

If you have the `eom` function, and optionally a function for the
Jacobian, you can use the constructor
```julia
ContinuousDS(state, eom! [, jacob! [, J]]; tspan = (0.0, 100.0))
```
with `state` the initial condition of the system.

If instead you already have an `ODEProblem` because you also want to take advantage
of the callback functionality of DifferentialEquations.jl, you may use the constructor
```julia
ContinuousDS(odeproblem [, jacob! [, J]])
```
If the `jacob!` is not provided by the user, it is created automatically
using the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/)
(which always passes `t=0` at the `eom!`) in both cases.

As mentioned in our [official documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/system_definition#example-using-functors),
it is preferred to use Functors for both the equations of motion and the Jacobian.

To interfece *towards* DifferentialEquations.jl use `ODEIntegrator(ds, stuff...)`.
Notice that you can have performance gains for stiff methods by
explicitly adding a Jacobian caller for DifferentialEquations.jl by defining
`eom!(::Type{Val{:jac}}, t, u, J) = jacob!(t, u, J)`.

`ContinuousDS` by default are evolved using solver `Vern9()` and tolerances
`:abstol => 1e-9, :reltol => 1e-9`.
"""
struct ContinuousDS{T<:Number, ODE<:ODEProblem, JJ} <: ContinuousDynamicalSystem
    prob::ODE
    jacob!::JJ
    J::Matrix{T}

    function ContinuousDS{T,ODE,JJ}(prob, j!, J) where {T,ODE,JJ}

        typeof(prob.u0) <: Vector || throw(ArgumentError(
        "Currently we only support vectors as states, "*
        "see the documentation string of `ContinuousDS`."
        ))
        j!(0, prob.u0, J)

        eltype(prob.u0) == eltype(J) || throw(ArgumentError(
        "The state and the Jacobian must have same type of numbers."))

        return new(prob, j!, J)
    end
end

# Constructors with Jacobian:
ContinuousDS(prob::ODE, j!::JJ, J::Matrix{T}) where {T<:Number, ODE<:ODEProblem, JJ} = ContinuousDS{T, ODE,JJ}(prob, j!, J)

function ContinuousDS(prob::ODEProblem, j!)

    J = zeros(eltype(state), length(state), length(state))
    return ContinuousDS(prob, j!, J)
end

function ContinuousDS(state, eom!, j!,
    J = zeros(eltype(state), length(state), length(state)); tspan=(0.0, 100.0))

    j!(0.0, state, J)
    problem = ODEProblem{true}(eom!, state, tspan)

    return ContinuousDS(problem, j!, J)
end


# Constructors without Jacobian:
function ContinuousDS(prob::ODEProblem)
    state = prob.u0
    eom! = prob.f

    D = length(state); T = eltype(state)
    du = copy(state)
    J = zeros(T, D, D)

    jeom! = (du, u) -> eom!(0, u, du)
    jcf = ForwardDiff.JacobianConfig(jeom!, du, state)
    ForwardDiff_jacob! = (t, u, J) -> ForwardDiff.jacobian!(
    J, jeom!, du, u, jcf)
    ForwardDiff_jacob!(0, state, J)

    return ContinuousDS(prob, ForwardDiff_jacob!, J)
end

function ContinuousDS(state, eom!; tspan=(0.0, 100.0))

    D = length(state); T = eltype(state)
    du = copy(state)
    J = zeros(T, D, D)

    problem = ODEProblem{true}(eom!, state, tspan)

    jeom! = (du, u) -> eom!(0, u, du)
    jcf = ForwardDiff.JacobianConfig(jeom!, du, state)
    ForwardDiff_jacob! = (t, u, J) -> ForwardDiff.jacobian!(
    J, jeom!, du, u, jcf)
    ForwardDiff_jacob!(0, state, J)

    return ContinuousDS(problem, ForwardDiff_jacob!, J)
end

# Basic
dimension(ds::ContinuousDS) = length(ds.prob.u0)
Base.eltype(ds::ContinuousDS{T,F,J}) where {T, F, J} = T
state(ds::ContinuousDS) = ds.prob.u0

jacobian(ds::ContinuousDynamicalSystem, t = 0) =
(ds.jacob!(t, state(ds), ds.J); ds.J)

#######################################################################################
#                         Interface to DifferentialEquations                          #
#######################################################################################

# ODEProblem helper functions
ODEProblem(ds::ContinuousDS) = ds.prob

ODEProblem(
ds::ContinuousDS, t::Real, state = ds.prob.u0) =
ODEProblem{true}(ds.prob.f, state, (zero(t), t),
callback = ds.prob.callback, mass_matrix = ds.prob.mass_matrix)

ODEProblem(ds::ContinuousDS, tspan::Tuple, state = ds.prob.u0) =
ODEProblem{true}(ds.prob.f, state, tspan,
callback = ds.prob.callback, mass_matrix = ds.prob.mass_matrix)

function ODEProblem(ds::ContinuousDS, t::Real, state, cb)
    if ds.prob.callback == nothing
        return ODEProblem{true}(ds.prob.f, state, (zero(t), t),
        callback = cb, mass_matrix = ds.prob.mass_matrix)
    else
        return ODEProblem{true}(ds.prob.f, state, (zero(t), t),
        callback = CallbackSet(cb, ds.prob.callback),
        mass_matrix = ds.prob.mass_matrix)
    end
end

"""
    ODEIntegrator(ds::ContinuousDS, t [, state]; diff_eq_kwargs)
Return an `ODEIntegrator` to be used directly with the interfaces of
[`DifferentialEquations.jl`](http://docs.juliadiffeq.org/stable/index.html).

`diff_eq_kwargs` is a dictionary `Dict{Symbol, ANY}`
of keyword arguments
passed into the `init` of
[`DifferentialEquations.jl`](http://docs.juliadiffeq.org/stable/index.html),
for example `Dict(:abstol => 1e-9)`. If you want to specify a solver,
do so by using the symbol `:solver`, e.g.:
`Dict(:solver => DP5(), :tstops => 0:0.01:t)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers.
"""
function OrdinaryDiffEq.ODEIntegrator(ds::ContinuousDS,
    t, state::Vector = ds.prob.u0; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    prob = ODEProblem(ds, t, state)
    solver, newkw = extract_solver(diff_eq_kwargs)
    integrator = init(prob, solver; newkw...,
    save_everystep=false)
    return integrator
end



"""
    variational_integrator(ds::ContinuousDS, k::Int, t, S::Matrix; diff_eq_kwargs)
Return an `ODEIntegrator` that represents the variational equations
of motion for the system. `t` makes the `tspan` and if it is `Real`
instead of `Tuple`, initial time is assumed zero.

This integrator evolves in parallel the system and `k` deviation
vectors ``w_i`` such that ``\\dot{w}_i = J\\times w_i`` with ``J`` the Jacobian
at the current state. `S` is the initial "conditions" which contain both the
system's state as well as the initial diviation vectors:
`S = cat(2, state, ws)` if `ws` is a matrix that has as *columns* the initial
deviation vectors.

The only keyword argument for this funcion is `diff_eq_kwargs` (see
[`trajectory`](@ref)).
"""
function variational_integrator(ds::ContinuousDS, k::Int, T,
    S::AbstractMatrix; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)

    f! = ds.prob.f
    jac! = ds.jacob!
    J = ds.J
    # the equations of motion `veom!` evolve the system and
    # k deviation vectors. Notice that the k deviation vectors
    # can also be considered a D×k matrix (which is the case
    # at `lyapunovs` function).
    # The e.o.m. for the system is f!(t, u , du) with `u` the system state.
    # The e.o.m. for the deviation vectors (tangent dynamics) are simply:
    # dY/dt = J(u) ⋅ Y
    # with J the Jacobian of the vector field at the current state
    # and Y being each of the k deviation vectors
    veom! = (t, u, du) -> begin
        us = view(u, :, 1)
        f!(t, us, view(du, :, 1))
        jac!(t, us, J)
        A_mul_B!(view(du, :, 2:k+1), J, view(u, :, 2:k+1))
    end

    if typeof(T) <: Real
        varprob = ODEProblem{true}(veom!, S, (zero(T), T))
    else
        varprob = ODEProblem{true}(veom!, S, T)
    end

    solver, newkw = extract_solver(diff_eq_kwargs)
    vintegrator = init(varprob, solver; newkw..., save_everystep=false)
    return vintegrator
end



function check_tolerances(d0, diff_eq_kwargs)
    defatol = 1e-6; defrtol = 1e-3
    atol = haskey(diff_eq_kwargs, :abstol) ? diff_eq_kwargs[:abstol] : defatol
    rtol = haskey(diff_eq_kwargs, :reltol) ? diff_eq_kwargs[:reltol] : defrtol
    if atol > 10d0
        warnstr = "Absolute tolerance (abstol) of integration is much larger than "
        warnstr*= "`d0`! It is highly suggested to decrease it using `diff_eq_kwargs`."
        warn(warnstr)
    end
    if rtol > 10d0
        warnstr = "Relative tolerance (reltol) of integration is much larger than "
        warnstr*= "`d0`! It is highly suggested to decrease it using `diff_eq_kwargs`."
        warn(warnstr)
    end
end
#######################################################################################
#                                Evolution of System                                  #
#######################################################################################
const DEFAULT_DIFFEQ_KWARGS = Dict{Symbol, Any}(:abstol => 1e-9, :reltol => 1e-9)
const DEFAULT_SOLVER = Vern9()

# See discrete.jl for the documentation string
function evolve(ds::ContinuousDS, t = 1.0, state = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    prob = ODEProblem(ds, t, state)
    return get_sol(prob, diff_eq_kwargs)[1][end]
end

evolve!(ds::ContinuousDS, t = 1.0; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS) =
(ds.prob.u0 .= evolve(ds, t, diff_eq_kwargs = diff_eq_kwargs))

function extract_solver(diff_eq_kwargs)
    # Extract solver from kwargs
    if haskey(diff_eq_kwargs, :solver)
        newkw = deepcopy(diff_eq_kwargs)
        solver = diff_eq_kwargs[:solver]
        pop!(newkw, :solver)
    else
        solver = DEFAULT_SOLVER
        newkw = diff_eq_kwargs
    end
    return solver, newkw
end

"""
    get_sol(prob::ODEProblem [, diff_eq_kwargs::Dict, extra_kwargs::Dict])
Solve the `prob` using `solve` and return the solutions vector as well as
the time vector.

The second and third
arguments are optional *position* arguments, passed to `solve` as keyword arguments.
They both have to be dictionaries of `Symbol` keys.
Only the second argument may contain a solver via the `:solver` key.

`get_sol` correctly uses `tstops` if necessary
(e.g. in the presence of `DiscreteCallback`s).
"""
function get_sol(prob::ODEProblem, diff_eq_kwargs::Dict = DEFAULT_DIFFEQ_KWARGS,
    extra_kwargs = Dict())

    solver, newkw = extract_solver(diff_eq_kwargs)
    # Take special care of callback sessions and use `tstops` if necessary
    # in conjuction with `saveat`
    if haskey(newkw, :saveat) && use_tstops(prob)
        sol = solve(prob, solver; newkw..., extra_kwargs..., save_everystep=false,
        tstops = newkw[:saveat])
    else
        sol = solve(prob, solver; newkw..., extra_kwargs..., save_everystep=false)
    end

    return sol.u, sol.t
end

function use_tstops(prob::ODEProblem)
    if prob.callback == nothing
        return false
    elseif typeof(prob.callback) <: CallbackSet
        any(x->typeof(x)<:DiscreteCallback, prob.callback.discrete_callbacks)
    else
        return typeof(prob.callback) <: DiscreteCallback
    end
end

# See discrete.jl for the documentation string
function trajectory(ds::ContinuousDS, T;
    dt::Real=0.05, diff_eq_kwargs::Dict = DEFAULT_DIFFEQ_KWARGS)

    # Necessary due to DifferentialEquations:
    if typeof(T) <: Real && !issubtype(typeof(T), AbstractFloat)
        T<=0 && throw(ArgumentError("Total time `T` must be positive."))
        T = convert(Float64, T)
    end

    if typeof(T) <: Real
        t = zero(T):dt:T #time vector
    elseif typeof(T) <: Tuple
        t = T[1]:dt:T[2]
    end

    prob = ODEProblem(ds, T)

    return Dataset(get_sol(prob, diff_eq_kwargs, Dict(:saveat => t))[1])
end

#######################################################################################
#                                 Pretty-Printing                                     #
#######################################################################################
Base.summary(ds::ContinuousDS) =
"$(dimension(ds))-dimensional continuous dynamical system"

function Base.show(io::IO, ds::ContinuousDS{S, F, J}) where {S, F, J}
    D = dimension(ds)
    text = summary(ds)
    print(io, text*":\n",
    "state: $(ds.prob.u0)\n", "e.o.m.: $(ds.prob.f)\n")
end
