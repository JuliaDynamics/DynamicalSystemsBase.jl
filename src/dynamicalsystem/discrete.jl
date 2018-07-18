using StaticArrays, ForwardDiff, DiffEqBase
using DiffEqBase: DEIntegrator
import DiffEqBase: init, step!, isinplace, reinit!, u_modified!
import Base: show

export MinimalDiscreteProblem, MinimalDiscreteIntegrator
export DiscreteDynamicalSystem, reinit!

#####################################################################################
#                          Discrete Dynamical System                                #
#####################################################################################
"""
    MinimalDiscreteProblem(eom, state, p = nothing, t0 = 0)

A minimal implementation of `DiscreteProblem` that is as fast as
possible. Does not save any points and cannot use callbacks.
"""
struct MinimalDiscreteProblem{IIP, S, D, F, P}
    f::F      # eom, but same syntax as ODEProblem
    u0::S     # initial state
    p::P      # parameter container
    t0::Int   # initial time
end
const MDP = MinimalDiscreteProblem
function MinimalDiscreteProblem(eom::F, state,
    p::P = nothing, t0 = 0) where {F, P}
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    if typeof(state) <: AbstractVector && eltype(state)<:Number
        IIP || typeof(eom(state, p, 0)) <: Union{SVector, Number} || error(
        "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    end
    u0 = safe_state_type(Val{IIP}(), state)
    S = typeof(u0)
    D = length(u0)
    MinimalDiscreteProblem{IIP, S, D, F, P}(eom, u0, p, t0)
end

isinplace(::MDP{IIP}) where {IIP} = IIP
systemtype(::MinimalDiscreteProblem) = "discrete"
inittime(prob::MDP) = prob.t0
dimension(::MinimalDiscreteProblem{IIP, S, D}) where {IIP, S, D} = D
get_state(prob::MDP) = prob.u0
stateeltype(::MDP{IIP, S}) where {IIP, S} = eltype(S)
@inline _get_eom(prob::MDP) = prob.f


"""
    DiscreteDynamicalSystem(eom, state, p [, jacobian [, J]]; t0::Int = 0)
A `DynamicalSystem` restricted to discrete-time systems (also called *maps*).
"""
struct DiscreteDynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD} <: DynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD}
    prob::MDP{IIP, S, D, F, P}
    jacobian::JAC
    J::JM
end
const DDS = DiscreteDynamicalSystem

function DiscreteDynamicalSystem(
    eom::F, s, p::P, j::JAC, J0::JM; t0::Int = 0) where {F, P, JAC, JM}
    if !(typeof(s) <: Union{AbstractVector, Number})
        throw(ArgumentError("
        The state of a dynamical system *must* be <: AbstractVector/Number!"))
    end
    prob = MDP(eom, s, p, t0)
    IIP = isinplace(prob)
    S = typeof(get_state(prob))
    D = length(s)
    return DDS{IIP, S, D, F, P, JAC, JM, false}(prob, j, J0)
end
function DiscreteDynamicalSystem(
    eom::F, s, p::P, j::JAC; t0::Int = 0)  where {F, P, JAC}
    if !(typeof(s) <: Union{AbstractVector, Number})
        throw(ArgumentError("
        The state of a dynamical system *must* be <: AbstractVector/Number!"))
    end
    prob = MDP(eom, s, p, t0)
    S = typeof(prob.u0)
    IIP = isinplace(prob)
    D = dimension(prob)
    J0 = get_J(prob, j)
    JM = typeof(J0)
    return DDS{IIP, S, D, F, P, JAC, JM, false}(prob, j, J0)
end
function DiscreteDynamicalSystem(
    eom::F, s, p::P; t0::Int = 0) where {F, P}
    if !(typeof(s) <: Union{AbstractVector, Number})
        throw(ArgumentError("
        The state of a dynamical system *must* be <: AbstractVector/Number!"))
    end
    prob = MDP(eom, s, p, t0)
    S = typeof(prob.u0)
    IIP = isinplace(prob)
    D = length(s)
    j = create_jacobian(eom, Val{IIP}(), prob.u0, p, t0, Val{D}())
    J0 = get_J(prob, j)
    JM = typeof(J0); JAC = typeof(j)
    return DDS{IIP, S, D, F, P, JAC, JM, true}(prob, j, J0)
end

timetype(::DDS) = Int

#####################################################################################
#                           MinimalDiscreteIntegrator                               #
#####################################################################################
mutable struct MinimalDiscreteIntegrator{IIP, S, D, F, P} <: DEIntegrator
    f::F      # integrator eom
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
    t0::Int   # initial time (only for reinit!)
end
const MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP
stateeltype(::MDI{IIP, S}) where {IIP, S} = eltype(S)
stateeltype(::MDI{IIP, S}) where {IIP, S<:Vector{<:AbstractArray{T}}} where {T} = T

function init(prob::MDP{IIP, S, D, F, P}, u0 = prob.u0) where {IIP, S, D, F, P}
    return MDI{IIP, S, D, F, P}(prob.f, S(u0), prob.t0, S(deepcopy(u0)), prob.p, prob.t0)
end

function reinit!(integ::MDI, u = integ.u; t0 = integ.t0, Q0 = nothing)
    integ.u = u
    integ.dummy = u
    integ.t = t0
    if Q0 != nothing
        set_deviations!(integ, Q0)
    end
    return
end

@inline function (integ::MinimalDiscreteIntegrator)(t::Real)
    if t == integ.t
        return integ.u
    else
        error("Can't extrapolate discrete systems!")
    end
end

# Get state for parallel:
get_state(a::MDI{IIP, S}) where {IIP, S<:Vector{<:AbstractVector}} = a.u[1]
get_state(a::MDI{IIP, S}, k) where {IIP, S<:Vector{<:AbstractVector}} = a.u[k]
function set_state!(
    integ::MDI{Alg, S}, u::AbstractVector, k::Int = 1
    ) where {Alg, S<:Vector{<:AbstractVector}}
    integ.u[k] = u
end


# for autodiffed in-place version
get_state(integ::MDI{Alg, S}) where {Alg, S<:AbstractMatrix} = integ.u[:, 1]
set_state!(integ::MDI{Alg, S}, u) where {Alg, S<:AbstractMatrix} = (integ.u[:, 1] .= u)
get_deviations(integ::MDI{Alg, S}) where {Alg, S<:Matrix} =
    @view integ.u[:, 2:end]
get_deviations(integ::MDI{Alg, S}) where {Alg, S<:SMatrix} =
    integ.u[:, 2:end]
set_deviations!(integ::MDI{Alg, S}, Q) where {Alg, S<:Matrix} =
    (integ.u[:, 2:end] = Q)
set_deviations!(integ::MDI{Alg, S}, Q) where {Alg, S<:SMatrix} =
    (integ.u = hcat(integ.u[:,1], Q))

#####################################################################################
#                                   Stepping                                        #
#####################################################################################
# IIP version
function step!(integ::MDI{true})
    # vector swap
    integ.dummy, integ.u = integ.u, integ.dummy
    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end
function step!(integ::MDI{true}, N)
    for i in 1:N
        integ.dummy, integ.u = integ.u, integ.dummy
        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
step!(integ::MDI{false}) =
(integ.u = integ.f(integ.u, integ.p, integ.t); integ.t +=1; nothing)
function step!(integ::MDI{false}, N)
    for i in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
end

u_modified!(integ::MDI, ::Bool) = nothing

step!(integ::MDI, N, stop_at_tdt) = step!(integ, N)

#####################################################################################
#                           TangentDiscreteIntegrator                               #
#####################################################################################

# For discrete systems a special, super-performant version of tangent integrator
# can exist. We make this here.

mutable struct TangentDiscreteIntegrator{IIP, S, D, F, P, JAC, JM, WM} <: DEIntegrator
    f::F            # integrator eom
    u::S            # integrator state
    t::Int          # integrator "time" (counter)
    dummy::S        # dummy, used only in the IIP version
    p::P            # parameter container, I don't know why
    t0::Int         # initial time (only for reinit!)
    jacobian::JAC   # jacobian function
    J::JM           # jacobian matrix
    W::WM           # tangent vectors (in form of matrix)
    dummyW::WM      # dummy, only used in IIP version
end

const TDI = TangentDiscreteIntegrator
stateeltype(::TDI{IIP, S}) where {IIP, S} = eltype(S)

u_modified!(t::TDI, a) = nothing

# set_state is same as in standard

get_deviations(t::TDI) = t.W
set_deviations!(t::TDI, Q) = (t.W = Q)

function reinit!(integ::TDI, u = integ.u;
    t0 = integ.t0, Q0 = nothing)
    set_state!(integ, u)
    if Q0 != nothing
        set_deviations!(integ, Q0)
    end
    integ.t = t0
    return
end


#####################################################################################
#                                 Tangent Stepping                                  #
#####################################################################################
function step!(integ::TDI{true})
    integ.dummy, integ.u = integ.u, integ.dummy
    integ.dummyW, integ.W = integ.W, integ.dummyW

    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.jacobian(integ.J, integ.u, integ.p, integ.t)

    mul!(integ.W, integ.J, integ.dummyW)
    integ.t += 1
    return
end
function step!(integ::TDI{true}, N::Real)
    for i in 1:N
        integ.dummy, integ.u = integ.u, integ.dummy
        integ.dummyW, integ.W = integ.W, integ.dummyW

        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.jacobian(integ.J, integ.u, integ.p, integ.t)

        mul!(integ.W, integ.J, integ.dummyW)
        integ.t += 1
    end
    return
end

function step!(integ::TDI{false})
    integ.u = integ.f(integ.u, integ.p, integ.t)
    J = integ.jacobian(integ.u, integ.p, integ.t)
    integ.W = J*integ.W
    integ.t += 1
    return
end
function step!(integ::TDI{false}, N::Real)
    for i in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        J = integ.jacobian(integ.u, integ.p, integ.t)
        integ.W = J*integ.W
        integ.t += 1
    end
    return
end

#####################################################################################
#                                 Integrators                                       #
#####################################################################################
function integrator(ds::DDS{IIP, S, D, F, P}, u0 = ds.prob.u0) where {IIP, S, D, F, P}
    return MinimalDiscreteIntegrator{IIP, S, D, F, P}(
    ds.prob.f, S(u0), ds.prob.t0, S(deepcopy(u0)), ds.prob.p, ds.prob.t0)
end

function tangent_integrator(ds::DDS, k::Int; kwargs...)
    return tangent_integrator(
    ds, orthonormal(dimension(ds), k); kwargs...)
end

function tangent_integrator(ds::DDS{IIP, S, D, F, P, JAC, JM}, Q0::AbstractMatrix;
    u0 = ds.prob.u0, t0 = inittime(ds)) where {IIP, S, D, F, P, JAC, JM}

    Q = safe_matrix_type(Val{IIP}(), Q0)
    s = safe_state_type(Val{IIP}(), u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    WM = typeof(Q)

    return TDI{IIP, S, D, F, P, JAC, JM, WM}(ds.prob.f, s, ds.prob.t0, deepcopy(s),
    ds.prob.p, ds.prob.t0, ds.jacobian::JAC, deepcopy(ds.J), Q, deepcopy(Q))
end

# Auto-diffed in-place version
function tangent_integrator(ds::DDS{true, S, D, F, P, JAC, JM, true},
    Q0::AbstractMatrix;
    u0 = ds.prob.u0, t0 = inittime(ds)) where {S, D, F, P, JAC, JM}

    R = D + length(Q0)
    k = size(Q0)[2]
    Q = safe_matrix_type(Val{IIP}(), Q0)
    u = safe_state_type(Val{IIP}(), u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent_iad(
        ds.prob.f, ds.J, u, ds.prob.p, t0, Val{k}())
    tanprob = MDP(tangentf, hcat(u, Q), ds.prob.p, t0)
    TF = typeof(tangentf)

    s = hcat(u, Q)
    SS = typeof(s)

    return MDI{true, SS, R, TF, P}(tangentf, s, t0, deepcopy(s), ds.prob.p, t0)
end

function parallel_integrator(ds::DDS, states)
    peom, st = create_parallel(ds, states)
    pprob = MDP(peom, st, ds.prob.p, ds.prob.t0)
    return init(pprob)
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(ds::DDS{IIP, S, D}, t, u = ds.prob.u0;
    dt::Int = 1, Ttr = 0) where {IIP, S, D}
    T = eltype(S)
    integ = integrator(ds, u)
    ti = inittime(ds)
    tvec = ti:dt:t+ti
    L = length(tvec)
    T = eltype(get_state(ds))
    data = Vector{SVector{D, T}}(undef, L)
    Ttr != 0 && step!(integ, Ttr)
    data[1] = integ.u
    for i in 2:L
        step!(integ, dt)
        data[i] = SVector{D, T}(integ.u)
    end
    return Dataset(data)
end

function trajectory(ds::DDS{false, S, 1}, t, u = ds.prob.u0;
	dt::Int = 1, Ttr = 0) where {S}
    ti = inittime(ds)
    tvec = ti:dt:t+ti
    L = length(tvec)
    integ = integrator(ds, u)
    data = Vector{S}(undef, L)
    Ttr != 0 && step!(integ, Ttr)
    data[1] = integ.u
    for i in 2:L
        step!(integ, dt)
        data[i] = integ.u
    end
    return data
end

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::MDP) =
"$(dimension(ds))-dimensional minimal discrete problem"

Base.summary(ds::MDI) =
"$(dimension(ds.prob))-dimensional minimal discrete integrator"

function Base.show(io::IO, ds::MDP)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(get_state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    )
end
function Base.show(io::IO, ds::MDI)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(get_state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    )
end
