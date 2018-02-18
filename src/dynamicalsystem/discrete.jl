using StaticArrays, ForwardDiff, DiffEqBase
import DiffEqBase: init, step!, isinplace, reinit!
import Base: show

export MinimalDiscreteProblem, MinimalDiscreteIntegrator
export DiscreteDynamicalSystem

#####################################################################################
#                          Minimal Discrete Problem                                 #
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
state(prob::MDP) = prob.u0

"""
    DiscreteDynamicalSystem(eom, state, p [, jacobian [, J]]; t0::Int = 0)
A `DynamicalSystem` restricted to discrete systems (also called *maps*).
"""
struct DiscreteDynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD} <: DynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD}
    prob::MDP{IIP, S, D, F, P}
    jacobian::JAC
    J::JM
end
const DDS = DiscreteDynamicalSystem

function get_J(prob::MDP{IIP, S, D}, jacob::JAC) where {IIP, S, D, JAC}
    if IIP
        J = similar(prob.u0, (D,D))
        jacob(J, prob.u0, prob.p, inittime(prob))
    else
        J = jacob(prob.u0, prob.p, inittime(prob))
    end
    return J
end

function DiscreteDynamicalSystem(
    eom::F, s, p::P, j::JAC, J0::JM; t0::Int = 0) where {F, P, JAC, JM}
    prob = MDP(eom, s, p, t0)
    IIP = isinplace(prob)
    S = typeof(state(prob))
    D = length(s)
    return DDS{IIP, S, D, F, P, JAC, JM, false}(prob, j, J0)
end
function DiscreteDynamicalSystem(
    eom::F, s, p::P, j::JAC; t0::Int = 0)  where {F, P, JAC}
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
    prob = MDP(eom, s, p, t0)
    S = typeof(prob.u0)
    IIP = isinplace(prob)
    D = length(s)
    j = create_jacobian(eom, Val{IIP}(), prob.u0, p, t0)
    J0 = get_J(prob, j)
    JM = typeof(J0); JAC = typeof(j)
    return DDS{IIP, S, D, F, P, JAC, JM, true}(prob, j, J0)
end


#####################################################################################
#                           MinimalDiscreteIntegrator                               #
#####################################################################################
mutable struct MinimalDiscreteIntegrator{IIP, S, D, F, P} <: DEIntegrator
    f::F      # integrator eom
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
end
const MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP

function init(prob::MDP{IIP, S, D, F, P}, u0 = prob.u0) where {IIP, S, D, F, P}
    return MDI{IIP, S, D, F, P}(prob.f, S(u0), prob.t0, S(deepcopy(u0)), prob.p)
end

function reinit!(integ::MDI, u = integ.prob.u0)
    integ.u = u
    integ.dummy = u
    integ.t = inittime(integ.prob)
    return
end

@inline function (integ::MinimalDiscreteIntegrator)(t::Real)
    if t == integ.t
        return t
    else
        error("Cant extrapolate discrete systems")
    end
end

#####################################################################################
#                                   Stepping                                        #
#####################################################################################
# IIP version
function step!(integ::MDI{true})
    # try vector swap
    integ.dummy, integ.u = integ.u, integ.dummy
    # integ.dummy .= integ.u
    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end
function step!(integ::MDI{true}, N::Int)
    for i in 1:N
        # try vector swap instead of # integ.dummy .= integ.u
        integ.dummy, integ.u = integ.u, integ.dummy
        # integ.dummy .= integ.u
        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
step!(integ::MDI{false}) =
(integ.u = integ.f(integ.u, integ.p, integ.t); integ.t +=1; nothing)
function step!(integ::MDI{false}, N::Int)
    for i in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
end

DiffEqBase.u_modified!(integ::MDI, ::Bool) = nothing


#####################################################################################
#                                 Integrators                                       #
#####################################################################################
function integrator(ds::DDS{IIP, S, D, F, P}, u0 = ds.prob.u0) where {IIP, S, D, F, P}
    return MinimalDiscreteIntegrator{IIP, S, D, F, P}(
    ds.prob.f, S(u0), ds.prob.t0, S(deepcopy(u0)), ds.prob.p)
end

function tangent_integrator(ds::DDS, k::Int; kwargs...)
    return tangent_integrator(
    ds, orthonormal(dimension(ds), k); kwargs...)
end

function tangent_integrator(ds::DDS{IIP, S, D, F, P}, Q0::AbstractMatrix;
    u0 = ds.prob.u0, t0 = inittime(ds)) where {IIP, S, D, F, P}

    R = D + length(Q0)
    k = size(Q0)[2]
    Q = safe_matrix_type(Val{IIP}(), Q0)
    u = safe_state_type(Val{IIP}(), u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent(ds.prob.f, ds.jacobian, ds.J, Val{IIP}(), Val{k}())
    tanprob = MDP(tangentf, hcat(u, Q), ds.prob.p, t0)
    TF = typeof(tangentf)

    s = hcat(u, Q)
    SS = typeof(s)

    return MDI{IIP, SS, R, TF, P}(tangentf, s, ds.prob.t0, deepcopy(s), ds.prob.p)
end

function parallel_integrator(ds::DDS, states; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    peom, st = create_parallel(ds, states)
    pprob = MDP(peom, st, ds.prob.p, ds.prob.t0)
    return init(pprob)
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(ds::DDS{IIP, S, D}, t, u = ds.prob.u0; dt::Int = 1) where {IIP, S, D}
    T = eltype(S)
    integ = integrator(ds, u)
    ti = inittime(ds)
    tvec = ti:dt:t
    L = length(tvec)
    T = eltype(state(ds))
    data = Vector{SVector{D, T}}(L)
    data[1] = u
    for i in 2:L
        step!(integ, dt)
        data[i] = SVector{D, T}(integ.u)
    end
    return Dataset(data)
end

function trajectory(ds::DDS{false, S, 1}, t, u, dt) where {S}
    ti = inittime(ds)
    tvec = ti:dt:t
    L = length(tvec)
    integ = integrator(ds, u)
    data = Vector{S}(L)
    data[1] = u
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
    rpad(" state: ", ps)*"$(state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    )
end
function Base.show(io::IO, ds::MDI)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    )
end
