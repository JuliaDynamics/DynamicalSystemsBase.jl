using OrdinaryDiffEq, StaticArrays

#####################################################################################
#                                    Auxilary                                       #
#####################################################################################
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

const DEFAULT_DIFFEQ_KWARGS = Dict{Symbol, Any}(:abstol => 1e-9, :reltol => 1e-9)
const DEFAULT_SOLVER = Vern9()
const CDS_TSPAN = (0.0, Inf)

#####################################################################################
#                           ContinuousDynamicalSystem                               #
#####################################################################################
"""
    ContinuousDynamicalSystem
Type-alias for a continuous `DynamicalSystem`.
"""
ContinuousDynamicalSystem{IIP, IAD, PT, JAC, JM} =
DynamicalSystem{IIP, IAD, PT, JAC, JM} where {IIP, IAD, PT<:ODEProblem, JAC, JM}

CDS = ContinuousDynamicalSystem

function ContinuousDynamicalSystem(eom, state::AbstractVector, p, j = nothing; t0=0.0,
    J0 = nothing)
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || typeof(eom(state, p, 0)) <: SVector || error(
    "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    prob = ODEProblem(eom, u0, CDS_TSPAN, p)
    if j == nothing
        return DS(prob)
    else
        return DS(prob, j; J0 = J0)
    end
end

function integrator(ds::CDS, u0 = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, saveat = nothing, tspan = CDS_TSPAN)
    solver, newkw = extract_solver(diff_eq_kwargs)
    prob = ODEProblem(ds.prob.f, ds.prob.u0, tspan, ds.prob.p; callback =
    ds.prob.callback, mass_matrix = ds.prob.mass_matrix)
    if saveat == nothing
        integ = init(prob, solver; newkw..., save_everystep = false)
    else
        integ = init(prob, solver; newkw..., saveat = saveat, save_everystep = false)
    end
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################

"""
```julia
trajectory(ds::DynamicalSystem, T [, u]; kwargs...) -> dataset
```
Return a dataset what will contain the trajectory of the sytem,
after evolving it for total time `T`, optionally starting from state `u`.
See [`Dataset`](@ref) for info on how to
manipulate this object.

For the discrete case, `T` is an integer and a `T×D` dataset is returned
(`D` is the system dimensionality). For the
continuous case, a `W×D` dataset is returned, with `W = length(t0:dt:T)` with
`t0:dt:T` representing the time vector (*not* returned).

## Keyword Arguments
* `dt = 0.01 | 1` :  Time step of value output during the solving
  of the continuous system. For discrete systems it must be an integer.
* `diff_eq_kwargs = Dict()` : (only for continuous) A dictionary `Dict{Symbol, ANY}`
  of keyword arguments
  passed into the solvers of the [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/basics/common_solver_opts.html)
  package, for example `Dict(:abstol => 1e-9)`. If you want to specify a solver,
  do so by using the symbol `:solver`, e.g.:
  `Dict(:solver => DP5(), :maxiters => 1e9)`. This requires you to have been first
  `using OrdinaryDiffEq` to access the solvers.
"""
function trajectory(ds::DynamicalSystem, T, u = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, dt = 0.01)

    tvec = inittime(ds):dt:(T+inittime(ds))
    tspan = (inittime(ds), inittime(ds) + T)
    integ = integrator(ds, u; tspan = tspan,
    diff_eq_kwargs = diff_eq_kwargs, saveat = tvec)
    solve!(integ)
    return Dataset(integ.sol.u)
end
