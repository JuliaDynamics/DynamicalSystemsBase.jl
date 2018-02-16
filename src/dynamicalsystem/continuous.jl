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
DynamicalSystem{IIP, IAD, PT, JAC, JM} where
{IIP, IAD, PT<:ODEProblem, JAC, JM}

CDS = ContinuousDynamicalSystem

function ContinuousDynamicalSystem(eom, state::AbstractVector, p, j = nothing; t0=0.0,
    J0 = nothing)
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || typeof(eom(state, p, 0)) <: SVector || error(
    "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    prob = ODEProblem(eom, u0, (Float64(t0), Inf), p)
    if j == nothing
        return DS(prob)
    else
        return DS(prob, j; J0 = J0)
    end
end

#####################################################################################
#                                 Integrators                                       #
#####################################################################################
function integrator(ds::CDS, u0 = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, saveat = nothing, tspan = ds.prob.tspan)
    solver, newkw = extract_solver(diff_eq_kwargs)
    prob = ODEProblem(ds.prob.f, ds.prob.u0, tspan, ds.prob.p; callback =
    ds.prob.callback, mass_matrix = ds.prob.mass_matrix)
    if saveat == nothing
        integ = init(prob, solver; newkw..., save_everystep = false)
    else
        integ = init(prob, solver; newkw..., saveat = saveat, save_everystep = false)
    end
end

function tangent_integrator(ds::CDS, k::Int;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    return tangent_integrator(
    ds, orthonormal(dimension(ds), k); u0 = u0; diff_eq_kwargs = diff_eq_kwargs)
end

function tangent_integrator(ds::CDS{IIP}, Q0::AbstractMatrix;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS) where {IIP}

    Q = safe_matrix_type(ds, Q0)
    u = safe_state_type(ds, u0)
    tangentf = create_tangent(ds)
    tanprob = ODEProblem{IIP}(tangentf, hcat(u, Q), (inittime(ds), Inf), ds.prob.p)

    solver, newkw = extract_solver(diff_eq_kwargs)
    return init(tanprob, solver; newkw..., save_everystep = false)
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(ds::DynamicalSystem, T, u = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, dt = 0.01)

    tvec = inittime(ds):dt:(T+inittime(ds))
    tspan = (inittime(ds), inittime(ds) + T)
    integ = integrator(ds, u; tspan = tspan,
    diff_eq_kwargs = diff_eq_kwargs, saveat = tvec)
    solve!(integ)
    return Dataset(integ.sol.u)
end
