using OrdinaryDiffEq, StaticArrays

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




function integrator(ds::CDS; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    solver, newkw = extract_solver(diff_eq_kwargs)
    integ = init(ds.prob, solver; newkw..., save_everystep = false)
end
