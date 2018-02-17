using OrdinaryDiffEq, StaticArrays

#####################################################################################
#                                    Auxilary                                       #
#####################################################################################
const DEFAULT_DIFFEQ_KWARGS = Dict(:abstol => 1e-9, :reltol => 1e-9)
const DEFAULT_SOLVER = Vern9()
const CDS_TSPAN = (0.0, Inf)

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

function ContinuousDynamicalSystem(eom, s::AbstractVector, p, j = nothing; t0=0.0,
    J0 = nothing)
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || typeof(eom(s, p, 0)) <: SVector || error(
    "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    u0 = IIP ? Vector(s) : SVector{length(s)}(s...)
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
function integrator(ds::CDS{iip}, u0 = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    saveat = nothing, tspan = ds.prob.tspan) where {iip}

    solver, newkw = extract_solver(diff_eq_kwargs)
    prob = ODEProblem{iip}(ds.prob.f, u0, tspan, ds.prob.p; callback =
    ds.prob.callback, mass_matrix = ds.prob.mass_matrix)
    if saveat == nothing
        integ = init(prob, solver; newkw..., save_everystep = false)
    else
        integ = init(prob, solver; newkw..., saveat = saveat, save_everystep = false)
    end
end

function tangent_integrator(ds::CDS, k::Int; kwargs...)
    return tangent_integrator(ds, orthonormal(dimension(ds), k); kwargs...)
end

function tangent_integrator(ds::CDS{IIP}, Q0::AbstractMatrix;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    t0 = inittime(ds)) where {IIP}

    Q = safe_matrix_type(ds, Q0)
    u = safe_state_type(ds, u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent(ds, size(Q)[2])
    tanprob = ODEProblem{IIP}(tangentf, hcat(u, Q), (t0, Inf), ds.prob.p)

    solver, newkw = extract_solver(diff_eq_kwargs)
    return init(tanprob, solver; newkw..., save_everystep = false)
end

# Vector-of-Vector does not work with DiffEq atm:
# This is a workaround currently, until DiffEq allows Vector[Vector]
function create_parallel(ds::CDS{true}, states)
    st = Matrix(hcat(states...))
    L = size(st)[2]
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            ds.prob.f(view(du, :, i), view(u, :, i), p, t)
        end
    end
    return paralleleom, st
end

function parallel_integrator(ds::CDS, states; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    peom, st = create_parallel(ds, states)
    pprob = ODEProblem(peom, st, (inittime(ds), Inf), ds.prob.p)
    solver, newkw = extract_solver(diff_eq_kwargs)
    return init(pprob, solver; newkw..., save_everystep = false)
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
