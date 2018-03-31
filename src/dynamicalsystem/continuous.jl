using OrdinaryDiffEq, StaticArrays
using OrdinaryDiffEq: ODEIntegrator

export ContinuousDynamicalSystem, CDS
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
    ContinuousDynamicalSystem(eom, state, p [, jacobian [, J]]; t0 = 0.0)
    A `DynamicalSystem` restricted to continuous systems (also called *flows*).
"""
struct ContinuousDynamicalSystem{
    IIP, S, D, F, P, JAC, JM, IAD, tType, JPROT, C, MM} <: DynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD}
    prob::ODEProblem{S, tType, IIP, P, F, JPROT, C, MM, DiffEqBase.StandardODEProblem}
    jacobian::JAC
    J::JM
end

const CDS = ContinuousDynamicalSystem
stateeltype(::CDS{IIP, S}) where {IIP, S} = eltype(S)
stateeltype(::ODEProblem{S}) where {S} = eltype(S)
timetype(::ContinuousDynamicalSystem{
IIP, S, D, F, P, JAC, JM, IAD, tType, JPROT, C, MM}) where
{IIP, S, D, F, P, JAC, JM, IAD, tType, JPROT, C, MM} = tType

function ContinuousDynamicalSystem(
    prob::ODEProblem{S, tType, IIP, P, F, JPROT, C, MM, DiffEqBase.StandardODEProblem},
    j::JAC, j0::JM, IAD::Bool) where {S, tType, IIP, P, F, JPROT, C, MM, JAC, JM}
    D = length(prob.u0)
    return ContinuousDynamicalSystem{
        IIP, S, D, F, P, JAC, JM, IAD, tType, JPROT, C, MM}(prob, j, j0)
end

# With jacobian:
function ContinuousDynamicalSystem(
    eom::F, s, p::P, j::JAC, J0::JM; t0 = 0.0, iad = false) where {F, P, JAC, JM}

    if !(typeof(s) <: AbstractVector)
        throw(ArgumentError("
        The state of a dynamical system *must* be <: AbstractVector in order
        to be able to create the tangent space."))
    end
    
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || typeof(eom(s, p, 0)) <: SVector || error(
    "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    u0 = safe_state_type(Val{IIP}(), s)

    prob = ODEProblem(eom, u0, (t0, oftype(t0, Inf)), p)
    return ContinuousDynamicalSystem(prob, j, J0, iad)
end
function ContinuousDynamicalSystem(
    eom::F, s, p::P, j::JAC; t0 = 0.0) where {F, P, JAC}
    J0 = get_J(j, s, p, t0)
    return ContinuousDynamicalSystem(eom, s, p, j, J0; t0 = t0)
end

# Without jacobian:
function ContinuousDynamicalSystem(
    eom::F, s, p::P; t0 = 0.0) where {F, P}
    IIP = isinplace(eom, 4)
    D = length(s)
    j = create_jacobian(eom, Val{IIP}(), s, p, t0, Val{D}())
    J0 = get_J(j, s, p, t0)
    return ContinuousDynamicalSystem(eom, s, p, j, J0; t0 = t0, iad = true)
end

#####################################################################################
#                                 Integrators                                       #
#####################################################################################
stateeltype(::ODEIntegrator{Alg, S}) where {Alg, S} = eltype(S)
stateeltype(::ODEIntegrator{Alg, S}) where {
    Alg, S<:Vector{<:AbstractArray{T}}} where {T} = T

function integrator(ds::CDS{iip}, u0 = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    saveat = nothing, tspan = ds.prob.tspan) where {iip}

    u = safe_state_type(Val{iip}(), u0)
    solver, newkw = extract_solver(diff_eq_kwargs)
    prob = ODEProblem{iip}(ds.prob.f, u, tspan, ds.prob.p)

    saveat != nothing && tspan[2] == Inf && error("Infinite solving!")

    if saveat == nothing
        integ = init(prob, solver; newkw..., save_everystep = false)
    else
        integ = init(prob, solver; newkw..., saveat = saveat, save_everystep = false)
    end
end

############################### Tangent ##############################################
function tangent_integrator(ds::CDS, k::Int; kwargs...)
    return tangent_integrator(ds, orthonormal(dimension(ds), k); kwargs...)
end
function tangent_integrator(ds::CDS{IIP}, Q0::AbstractMatrix;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    t0 = inittime(ds)) where {IIP}

    Q = safe_matrix_type(Val{IIP}(), Q0)
    u = safe_state_type(Val{IIP}(), u0)

    k = size(Q)[2]
    k > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent(ds.prob.f, ds.jacobian, ds.J, Val{IIP}(), Val{k}())
    tanprob = ODEProblem{IIP}(tangentf, hcat(u, Q), (t0, Inf), ds.prob.p)

    solver, newkw = extract_solver(diff_eq_kwargs)
    return init(tanprob, solver; newkw..., save_everystep = false)
end

# Auto-diffed in-place version
function tangent_integrator(ds::CDS{true, S, D, F, P, JAC, JM, true},
    Q0::AbstractMatrix;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS,
    t0 = inittime(ds)) where {S, D, F, P, JAC, JM}

    Q = safe_matrix_type(Val{true}(), Q0)
    u = safe_state_type(Val{true}(), u0)

    k = size(Q)[2]
    k > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent_iad(
        ds.prob.f, ds.J, u, ds.prob.p, t0, Val{k}())
    tanprob = ODEProblem{true}(tangentf, hcat(u, Q), (t0, Inf), ds.prob.p)

    solver, newkw = extract_solver(diff_eq_kwargs)
    return init(tanprob, solver; newkw..., save_everystep = false)
end



############################### Parallel ##############################################
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

#####################################################################################
#                                    Get States                                     #
#####################################################################################
get_state(integ::ODEIntegrator{Alg, S}) where {Alg, S<:AbstractVector} = integ.u
get_state(integ::ODEIntegrator{Alg, S}) where {Alg, S<:AbstractMatrix} = integ.u[:, 1]
get_state(integ::ODEIntegrator{Alg, S}) where {Alg, S<:Vector{<:AbstractVector}} =
    integ.u[1]

function set_state!(
    integ::ODEIntegrator{Alg, S}, u::AbstractVector, k::Int = 1
    ) where {Alg, S<:Vector{<:AbstractVector}}
    integ.u[k] = u
end
function set_state!(
    integ::ODEIntegrator{Alg, S}, u::AbstractVector, k::Int = 1
    ) where {Alg, S<:AbstractMatrix}
    integ.u[:, k] .= u
end

get_tangent(integ::ODEIntegrator{Alg, S}) where {Alg, S<:AbstractVector} =
    error("It has no tangent dude")
get_tangent(integ::ODEIntegrator{Alg, S}) where {Alg, S<:Matrix} =
    @view integ.u[:, 2:end]
get_tangent(integ::ODEIntegrator{Alg, S}) where {Alg, S<:SMatrix} =
    integ.u[:, 2:end]

set_tangent!(integ::ODEIntegrator{Alg, S}, Q) where {Alg, S<:Matrix} =
    (integ.u[:, 2:end] .= Q)
set_tangent!(integ::ODEIntegrator{Alg, S}, Q) where {Alg, S<:SMatrix} =
    (integ.u = hcat(integ.u[:,1], Q))
