using OrdinaryDiffEq, Requires, ForwardDiff
import OrdinaryDiffEq.ODEProblem
import OrdinaryDiffEq.ODEIntegrator

export ContinuousDS, variational_integrator, ODEIntegrator
export ContinuousDynamicalSystem

#######################################################################################
#                                     Constructors                                    #
#######################################################################################
"Abstract type representing continuous systems."
abstract type ContinuousDynamicalSystem <: DynamicalSystem end

"""
    ContinuousDS(state, eom! [, jacob! [, J]]) <: DynamicalSystem
`D`-dimensional continuous dynamical system.
## Fields
* `state::Vector{T}` : Current state-vector of the system. Do `ds.state .= u` to
  change the state.
* `problem::ODEProblem` : The fundamental structure used in the
   [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/index.html).
   Contains the equations of motion, callbacks and other information.
* `jacob!` (function) : The function that represents the Jacobian of the system,
  given in the format: `jacob!(t, u, J)` which means it is in-place, with the mutated
  argument being the last.
* `J::Matrix{T}` : Jacobian matrix.

## Creating a `ContinuousDS`
The equations of motion should be in the form `eom!(t, u, du)`
which means that it is **in-place**, with the mutated argument
`du` the last one. If you have this function, and optionally a function for the
Jacobian, you can use the constructor
```julia
ContinuousDS(state, eom! [, jacob! [, J]]; tspan = (0.0, 100.0))
```
If instead you already have an `ODEProblem` because you also want to take advantage
of the callback functionality of DifferentialEquations.jl, you may use the constructor
```julia
ContinuousDS(odeproblem [, jacob! [, J]])
```

As mentioned in our [official documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/system_definition#example-using-functors),
it is preferred to use Functors for both the equations of motion and the Jacobian.

If the `jacob` is not provided by the user, it is created automatically
using the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/)
(which always passes `t=0` at the `eom!`).
Please see the official documentation for the handling of explicit time dependence
regarding Jacobians.

To interfece *towards* DifferentialEquations.jl use `ODEIntegrator(ds, stuff...)`.
"""
struct ContinuousDS{T<:Number, ODE, JJ} <: ContinuousDynamicalSystem
    state::Vector{T}
    problem::ODE
    jacob!::JJ
    J::Matrix{T}
end

# Constructors
function ContinuousDS(state, eom!, j!,
    J = zeros(eltype(state), length(state), length(state)); tspan=(0.0, 100.0))

    j!(0.0, state, J)
    problem = ODEProblem{true}(eom!, state, tspan)

    return ContinuousDS(state, problem, j!, J)
end

function ContinuousDS(state, eom!)
    D = length(state); T = eltype(state)
    du = copy(state)
    J = zeros(T, D, D)
    jeom! = (du, u) -> eom!(0, u, du)
    jcf = ForwardDiff.JacobianConfig(jeom!, du, state)
    ForwardDiff_jacob!(t, u, J) = ForwardDiff.jacobian!(
    J, jeom!, du, u, jcf)
    ForwardDiff_jacob!(0, state, J)
    return ContinuousDS(state, eom!, ForwardDiff_jacob!, J)
end

dimension(ds::ContinuousDS) = length(ds.state)
Base.eltype(ds::ContinuousDS{T,F,J}) where {T, F, J} = T

#######################################################################################
#                         Interface to DifferentialEquations                          #
#######################################################################################
function get_solver(diff_eq_kwargs::Dict)

    if haskey(diff_eq_kwargs, :solver)
        newkw = deepcopy(diff_eq_kwargs)
        solver = diff_eq_kwargs[:solver]
        pop!(newkw, :solver)
        return solver, newkw
    else
        solver = Tsit5()
        return solver, diff_eq_kwargs
    end
end

OrdinaryDiffEq.ODEProblem(
ds::ContinuousDS, t::Real = ds.problem.tspan[2], state::Vector = ds.state) =
ODEProblem(ds.problem, state, (zero(t), t))

OrdinaryDiffEq.ODEProblem(
ds::ContinuousDS, tspan::Tuple = ds.problem.tspan, state::Vector = ds.state) =
ODEProblem(ds.problem, state, tspan)

"""
    ODEIntegrator(ds::ContinuousDS, t [, state]; diff_eq_kwargs = Dict())
Return an `ODEIntegrator` to be used directly with the interfaces of
[`DifferentialEquations.jl`](http://docs.juliadiffeq.org/stable/index.html).

`diff_eq_kwargs = Dict()` is a dictionary `Dict{Symbol, ANY}`
of keyword arguments
passed into the `init` of
[`DifferentialEquations.jl`](http://docs.juliadiffeq.org/stable/index.html),
for example `Dict(:abstol => 1e-9)`. If you want to specify a solver,
do so by using the symbol `:solver`, e.g.:
`Dict(:solver => DP5(), :tstops => 0:0.01:t)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers.
"""
function OrdinaryDiffEq.ODEIntegrator(ds::ContinuousDS,
    t, state::Vector = ds.state; diff_eq_kwargs = Dict())
    prob = ODEProblem(ds, t, state)
    solver, newkw = get_solver(diff_eq_kwargs)
    integrator = init(prob, solver; newkw...,
    save_everystep=false)
    return integrator
end



"""
    get_sol(prob::ODEProblem, diff_eq_kwargs::Dict = Dict())
Solve the `prob` using `solve` and return the solution.
"""
function get_sol(prob::ODEProblem, diff_eq_kwargs::Dict = Dict())
    solver, newkw = get_solver(diff_eq_kwargs)
    sol = solve(prob, solver; newkw..., save_everystep=false)
    return sol.u
end


"""
    variational_integrator(ds::ContinuousDS, k::Int, tfinal, S::Matrix, kwargs...)
Return an `ODEIntegrator` that represents the variational equations
of motion for the system.

It evolves in parallel `ds.state` and `k` deviation
vectors ``w_i`` such that ``\\dot{w}_i = J\\times w_i`` with ``J`` the Jacobian
at the current state. `S` is the initial "conditions" which contain both the
system's state as well as the initial diviation vectors:
`S = cat(2, ds.state, ws)` if `ws` is a matrix that has as *columns* the initial
deviation vectors.

The only keyword argument for this funcion is `diff_eq_kwargs = Dict()` (see
[`trajectory`](@ref)).
"""
function variational_integrator(ds::ContinuousDS, k::Int, t_final::Real,
    S::AbstractMatrix; diff_eq_kwargs = Dict())

    f! = ds.eom!
    jac! = ds.jacob!
    J = ds.J
    # the equations of motion `veom!` evolve the system and
    # k deviation vectors. Notice that the k deviation vectors
    # can also be considered a D×k matrix (which is the case
    # at `lyapunovs` function).
    # The e.o.m. for the system is f!(t, u , du).
    # The e.o.m. for the deviation vectors (tangent dynamics) are simply:
    # dY/dt = J(u) ⋅ Y
    # with J the Jacobian of the vector field at the current state
    # and Y being each of the k deviation vectors
    veom! = (t, u, du) -> begin
        us = view(u, :, 1)
        f!(view(du, :, 1), us)
        jac!(J, us)
        A_mul_B!(view(du, :, 2:k+1), J, view(u, :, 2:k+1))
    end

    varprob = ODEProblem(veom!, S, (zero(t_final), t_final))
    solver, newkw = get_solver(diff_eq_kwargs)
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
# See discrete.jl for the documentation string
function evolve(ds::ContinuousDS, t::Real = 1.0, state = ds.state;
    diff_eq_kwargs = Dict())
    prob = ODEProblem(ds, t, state)
    return get_sol(prob, diff_eq_kwargs)[end]
end



# See discrete.jl for the documentation string
function trajectory(ds::ContinuousDS, T::Real;
    dt::Real=0.05, diff_eq_kwargs = Dict())

    # Necessary due to DifferentialEquations:
    if !issubtype(typeof(T), AbstractFloat)
        T = convert(Float64, T)
    end
    T<=0 && throw(ArgumentError("Total time `T` must be positive."))

    D = dimension(ds)
    t = zero(T):dt:T #time vector
    prob = ODEProblem(ds, T)
    kw = Dict{Symbol, Any}(diff_eq_kwargs) #nessesary conversion to add :saveat
    kw[:saveat] = t
    return Dataset(get_sol(prob, kw))
end

#######################################################################################
#                                 Pretty-Printing                                     #
#######################################################################################
function Base.show(io::IO, ds::ContinuousDS{S, F, J}) where {S, F, J}
    D = dimension(ds)
    text = "$(dimension(ds))-dimensional continuous dynamical system"
    print(io, text*":\n",
    "state: $(ds.state)\n", "eom: $(ds.problem.f)\n")
end
