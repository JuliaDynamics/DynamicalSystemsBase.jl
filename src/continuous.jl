using DiffEqBase, StaticArrays
using DiffEqBase: __init, ODEFunction, AbstractODEIntegrator

export CDS_KWARGS
#####################################################################################
#                                    Defaults                                       #
#####################################################################################
using SimpleDiffEq: SimpleATsit5
const DEFAULT_SOLVER = SimpleATsit5()
const DEFAULT_DIFFEQ_KWARGS = (abstol = 1e-6, reltol = 1e-6)
const CDS_KWARGS = (alg = DEFAULT_SOLVER, DEFAULT_DIFFEQ_KWARGS...)

_get_solver(a) = haskey(a, :alg) ? a[:alg] : DEFAULT_SOLVER

#####################################################################################
#                               Interface to DiffEq                                 #
#####################################################################################
function ContinuousDynamicalSystem(prob::ODEProblem, args...)
    return ContinuousDynamicalSystem(prob.f.f, prob.u0, prob.p, args...;
           t0 = prob.tspan[1])
end

function DiffEqBase.ODEProblem(ds::CDS{IIP}, tspan, args...) where {IIP}
    return ODEProblem{IIP}(ODEFunction(ds.f; jac = ds.jacobian), ds.u0, tspan, args...)
end

#####################################################################################
#                                 Integrators                                       #
#####################################################################################
stateeltype(::AbstractODEIntegrator{A, IIP, S}) where {A, IIP, S} = eltype(S)
stateeltype(::AbstractODEIntegrator{A, IIP, S}) where {
    A, IIP, S<:Vector{<:AbstractArray{T}}} where {T} = T

function integrator(ds::CDS{iip}, u0 = ds.u0;
    tfinal = Inf, diffeq...) where {iip}

    u = safe_state_type(Val{iip}(), u0)
    prob = ODEProblem{iip}(ds.f, u, (ds.t0, typeof(ds.t0)(tfinal)), ds.p)

    solver = _get_solver(diffeq)
    integ = __init(prob, solver; DEFAULT_DIFFEQ_KWARGS...,
                   save_everystep = false, diffeq...)
    return integ
end

############################### Tangent #############################################
function tangent_integrator(ds::CDS, k::Int = dimension(ds); kwargs...)
    return tangent_integrator(ds, orthonormal(dimension(ds), k); kwargs...)
end
function tangent_integrator(ds::CDS{IIP}, Q0::AbstractMatrix;
    u0 = ds.u0, diffeq...) where {IIP}

    t0 = ds.t0
    Q = safe_matrix_type(Val{IIP}(), Q0)
    u = safe_state_type(Val{IIP}(), u0)

    k = size(Q)[2]
    k > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent(ds.f, ds.jacobian, ds.J, Val{IIP}(), Val{k}())
    tanprob = ODEProblem{IIP}(tangentf, hcat(u, Q), (t0, typeof(t0)(Inf)), ds.p)

    solver = _get_solver(diffeq)
    return __init(tanprob, solver; DEFAULT_DIFFEQ_KWARGS..., internalnorm = _tannorm,
                  save_everystep = false, diffeq...)
end

function _tannorm(u::AbstractMatrix, t)
    s = size(u)[1]
    x = zero(eltype(u))
    for i in 1:s
        @inbounds x += u[i, 1]^2
    end
    return sqrt(x/length(x))
end
_tannorm(u::Real, t) = abs(u)

# Auto-diffed in-place version
function tangent_integrator(ds::CDS{true, S, D, F, P, JAC, JM, true},
    Q0::AbstractMatrix;
    u0 = ds.u0, diffeq...) where {S, D, F, P, JAC, JM}

    t0 = ds.t0
    Q = safe_matrix_type(Val{true}(), Q0)
    u = safe_state_type(Val{true}(), u0)

    k = size(Q)[2]
    k > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent_iad(ds.f, ds.J, u, ds.p, t0, Val{k}())
    tanprob = ODEProblem{true}(tangentf, hcat(u, Q), (t0, typeof(t0)(Inf)), ds.p)

    solver = _get_solver(diffeq)
    return __init(tanprob, solver; DEFAULT_DIFFEQ_KWARGS..., save_everystep = false,
                  internalnorm = _tannorm, diffeq...)
end

############################### Parallel ############################################
# Vector-of-Vector does not work with DiffEq atm:
# This is a workaround currently, until DiffEq allows Vector[Vector]
function create_parallel(ds::CDS{true}, states)
    st = Matrix(hcat(states...))
    L = size(st)[2]
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            ds.f(view(du, :, i), view(u, :, i), p, t)
        end
    end
    return paralleleom, st
end

# const STIFFSOLVERS = (ImplicitEuler, ImplicitMidpoint, Trapezoid, TRBDF2,
# GenericImplicitEuler,
# GenericTrapezoid, SDIRK2, Kvaerno3, KenCarp3, Cash4, Hairer4, Hairer42, Kvaerno4,
# KenCarp4, Kvaerno5, KenCarp5, Rosenbrock23,
# Rosenbrock32, ROS3P, Rodas3, RosShamp4, Veldd4, Velds4, GRK4T,
# GRK4A, Ros4LStab, Rodas4, Rodas42, Rodas4P)

function parallel_integrator(ds::CDS, states; diffeq...)
    peom, st = create_parallel(ds, states)
    pprob = ODEProblem(peom, st, (ds.t0, typeof(ds.t0)(Inf)), ds.p)
    solver = _get_solver(diffeq)
    # if typeof(solver) ∈ STIFFSOLVERS
    #     error("Stiff solvers can't support a parallel integrator.")
    # end
    if !(typeof(ds) <: CDS{true})
        return __init(pprob, solver; DEFAULT_DIFFEQ_KWARGS..., save_everystep = false,
                      internalnorm = _parallelnorm, diffeq...)
    else
        return __init(pprob, solver; DEFAULT_DIFFEQ_KWARGS..., save_everystep = false,
                      internalnorm = _tannorm, diffeq...)
    end
end

_parallelnorm(u::AbstractVector, t) = @inbounds DiffEqBase.ODE_DEFAULT_NORM(u[1], t)
_parallelnorm(u::Real, t) = abs(u)

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(ds::ContinuousDynamicalSystem, T, u = ds.u0;
    dt = 0.01, Ttr = 0.0, diffeq...)

    t0 = ds.t0
    tvec = (t0+Ttr):dt:(T+t0+Ttr)
    sol = Vector{SVector{dimension(ds), stateeltype(ds)}}(undef, length(tvec))
    integ = integrator(ds, u; dt = dt, diffeq...)
    step!(integ, Ttr)
    for (i, t) in enumerate(tvec)
        while t > integ.t
            step!(integ)
        end
        if integ.tprev ≤ t ≤ integ.t
            sol[i] = integ(t)
        else
            error("should be integ.tprev ≤ t ≤ integ.t")
        end
    end
    return Dataset(sol)
end

#####################################################################################
#                                    Get States                                     #
#####################################################################################
get_state(integ::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:AbstractVector} =
    integ.u
get_state(integ::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:AbstractMatrix} =
    integ.u[:, 1]
get_state(integ::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:Vector{<:AbstractVector}} =
    integ.u[1]
get_state(integ::AbstractODEIntegrator{Alg, IIP, S}, k::Int) where {Alg, IIP, S<:Vector{<:AbstractVector}} =
    integ.u[k]
get_state(integ::AbstractODEIntegrator{Alg, IIP, S}, k::Int) where {Alg, IIP, S<:AbstractMatrix} =
    integ.u[:, k]

function set_state!(
    integ::AbstractODEIntegrator{Alg, IIP, S}, u::AbstractVector, k::Int = 1
    ) where {Alg, IIP, S<:Vector{<:AbstractVector}}
    integ.u[k] = u
    u_modified!(integ, true)
end
function set_state!(
    integ::AbstractODEIntegrator{Alg, IIP, S}, u::AbstractVector
    ) where {Alg, IIP, S<:Matrix}
    integ.u[:, 1] .= u
    u_modified!(integ, true)
end
function set_state!(
    integ::AbstractODEIntegrator{Alg, IIP, S}, u::AbstractVector
    ) where {Alg, IIP, S<:SMatrix{D, K}} where {D, K}
    integ.u = hcat(SVector{D}(u), integ.u[:, SVector{K-1}(2:K...)])
    u_modified!(integ, true)
end

get_deviations(integ::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:Matrix} =
    @view integ.u[:, 2:end]


@generated function get_deviations(
    integ::AbstractODEIntegrator{Alg, IIP, S}) where {Alg, IIP, S<:SMatrix{D,K}} where {D,K}
    gens = [:($k) for k=2:K]
    quote
        sind = SVector{$(K-1)}($(gens...))
        integ.u[:, sind]
    end
end

set_deviations!(integ::AbstractODEIntegrator{Alg, IIP, S}, Q) where {Alg, IIP, S<:Matrix} =
    (integ.u[:, 2:end] .= Q; u_modified!(integ, true))
set_deviations!(integ::AbstractODEIntegrator{Alg, IIP, S}, Q) where {Alg, IIP, S<:SMatrix} =
    (integ.u = hcat(integ.u[:,1], Q); u_modified!(integ, true))

function DiffEqBase.reinit!(integ::AbstractODEIntegrator, u0::AbstractVector,
    Q0::AbstractMatrix; kwargs...)

    set_state!(integ, u0)
    set_deviations!(integ, Q0)
    reinit!(integ, integ.u; kwargs...)
end
