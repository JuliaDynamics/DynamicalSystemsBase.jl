using StaticArrays, ForwardDiff, DiffEqBase
using DiffEqBase: isinplace
using OrdinaryDiffEq: FunctionMap
import DiffEqBase: init, step!, isinplace, reinit!
import Base: show

export MinimalDiscreteProblem, MinimalDiscreteIntegrator
export DiscreteDynamicalSystem

const DDS_TSPAN = (0, Int(1e6))

#####################################################################################
#                          Minimal Discrete Problem                                 #
#####################################################################################
"""
    MinimalDiscreteProblem(eom, state, p = nothing, t0 = 0)

A minimal implementation of `DiscreteProblem` that is as fast as
possible. Does not save any points and cannot use callbacks.
"""
struct MinimalDiscreteProblem{IIP, F, S, P} <: DEProblem
    f::F      # eom, but same syntax as ODEProblem
    u0::S     # initial state
    p::P      # parameter container
    t0::Int   # initial time
end
MDP = MinimalDiscreteProblem
function MinimalDiscreteProblem(eom::F, state::AbstractArray{X},
    p::P = nothing, t0 = 0) where {F, P, X}
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    if typeof(state) <: AbstractVector && X<:Number
        IIP || typeof(eom(state, p, 0)) <: Union{SVector, Number} || error(
        "Equations of motion must return an `SVector` for DynamicalSystems.jl")
        u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    else
        u0 = state
    end
    S = typeof(u0)
    D = length(u0);
    MinimalDiscreteProblem{IIP, F, S, P}(eom, u0, p, t0)
end

isinplace(::MDP{IIP}) where {IIP} = IIP
systemtype(::MinimalDiscreteProblem) = "discrete"
problemtype(ds::DS{IIP, IAD, DEP, JAC, JM}) where
{DEP<:MinimalDiscreteProblem, IIP, JAC, IAD, JM} = MinimalDiscreteProblem
inittime(prob::MDP) = prob.t0

"""
    DiscreteDynamicalSystem(eom, state, p [, jacobian]; t0::Int = 0)
A `DynamicalSystem` restricted to discrete systems (also called *maps*).

Relevant functions:
"""
DiscreteDynamicalSystem{IIP, IAD, PT, JAC, JM} =
DynamicalSystem{IIP, IAD, PT, JAC, JM} where
{IIP, IAD, PT<:Union{DiscreteProblem, MinimalDiscreteProblem}, JAC, JM}

DDS = DiscreteDynamicalSystem

function DiscreteDynamicalSystem(
    eom, state, p, j = nothing; t0::Int = 0, J0 = nothing)
    prob = MDP(eom, state, p, t0)
    if j == nothing
        return DS(prob)
    else
        return DS(prob, j; J0 = J0)
    end
end

#####################################################################################
#                           MinimalDiscreteIntegrator                               #
#####################################################################################
mutable struct MinimalDiscreteIntegrator{IIP, F, S, P} <: DEIntegrator
    prob::MDP{IIP, F, S, P}
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
end
MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP

init(prob::MDP, ::FunctionMap, u0::AbstractVector) = init(prob, u0)
function init(prob::MDP{IIP, F, S, P}, u::AbstractVector) where
    {IIP, F, S, P}
    u0 = IIP ? Vector(u) : SVector{length(u)}(u...)
    return MDI{IIP, F, S, P}(prob, u0, prob.t0, deepcopy(u0), prob.p)
end
function init(prob::MDP{IIP, F, S, P}) where {IIP, F, S, P}
    return MDI{IIP, F, S, P}(prob, prob.u0, prob.t0, deepcopy(prob.u0), prob.p)
end

function reinit!(integ::MDI, u = integ.prob.u0)
    integ.u = u
    integ.dummy = u
    integ.t = inittime(integ.prob)
    return
end

#####################################################################################
#                                   Stepping                                        #
#####################################################################################
# IIP version
function step!(integ::MDI{true})
    integ.dummy .= integ.u
    integ.prob.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end
function step!(integ::MDI{true}, N::Int)
    for i in 1:N
        integ.dummy .= integ.u
        integ.prob.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
step!(integ::MDI{false}) =
(integ.u = integ.prob.f(integ.u, integ.p, integ.t); integ.t +=1; nothing)
function step!(integ::MDI{false}, N::Int)
    for i in 1:N
        integ.u = integ.prob.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
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

#####################################################################################
#                                 Integrators                                       #
#####################################################################################
function integrator(ds::DDS, u0::AbstractVector = ds.prob.u0)
    U0 = safe_state_type(ds, u0)
    if typeof(ds.prob) <: DiscreteProblem
        prob = DiscreteProblem(ds.prob.f, U0, DDS_TSPAN, ds.prob.p;
        callback = ds.prob.callback)
        integ = init(prob, FunctionMap(); save_everystep = false)
    elseif typeof(ds.prob) <: MDP
        integ = init(ds.prob, U0)
    else
        error("Unknown Discrete system Problem Type.")
    end
    return integ
end

function tangent_integrator(ds::DDS, k::Int;
    u0 = ds.prob.u0)
    return tangent_integrator(
    ds, orthonormal(dimension(ds), k); u0 = u0)
end

function tangent_integrator(ds::DDS{IIP}, Q0::AbstractMatrix;
    u0 = ds.prob.u0) where {IIP}

    Q = safe_matrix_type(ds, Q0)
    u = safe_state_type(ds, u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent(ds, size(Q)[2])
    tanprob = MDP(tangentf, hcat(u, Q), ds.prob.p, inittime(ds))
    return init(tanprob)
end

function parallel_integrator(ds::DDS, states; diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)
    peom, st = create_parallel(ds, states)
    pprob = MDP(peom, st, ds.prob.p, ds.prob.t0)
    return init(pprob)
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(ds::DDS, t, u = ds.prob.u0; dt::Int = 1)

    D = dimension(ds); T = eltype(ds)
    integ = integrator(ds, u)
    ti = inittime(ds)
    tvec = ti:dt:t
    L = length(tvec)
    data = Vector{SVector{D, T}}(L)
    data[1] = u
    for i in 2:L
        step!(integ, dt)
        data[i] = integ.u
    end

    return Dataset(data)
end
