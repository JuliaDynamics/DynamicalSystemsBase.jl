using StaticArrays, ForwardDiff, DiffEqBase
using DiffEqBase: isinplace
using OrdinaryDiffEq: FunctionMap
import DiffEqBase: init, step!, isinplace
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
struct MinimalDiscreteProblem{IIP, F, S, P, D, T} <: DEProblem
    # D, T are dimension and eltype of state
    f::F      # eom, but same syntax as ODEProblem
    u0::S     # initial state
    p::P      # parameter container
    t0::Int   # initial time
end
MDP = MinimalDiscreteProblem
function MinimalDiscreteProblem(eom::F, state, p::P = nothing, t0 = 0) where {F, P}
    IIP = isinplace(eom, 4)
    # Ensure that there are only 2 cases: OOP with SVector or IIP with Vector
    # (requirement from ChaosTools)
    IIP || typeof(eom(state, p, 0)) <: Union{SVector, Number} || error(
    "Equations of motion must return an `SVector` for DynamicalSystems.jl")
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    S = typeof(u0)
    D = length(u0); T = eltype(u0)
    D == 1 && (u0 = u0[1]) # handle 1D case
    MinimalDiscreteProblem{IIP, F, S, P, D, T}(eom, u0, p, t0)
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
#                                 integrator                                        #
#####################################################################################
mutable struct MinimalDiscreteIntegrator{IIP, F, S, P, D, T} <: DEIntegrator
    prob::MDP{IIP, F, S, P, D, T}
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
end
MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP

init(prob::MDP, ::FunctionMap, u0::AbstractVector) = init(prob, u0)
function init(prob::MDP{IIP, F, S, P, D, T}, u::AbstractVector = prob.u0
    ) where {IIP, F, S, P, D, T}
    u0 = IIP ? Vector(u) : SVector{length(u)}(u...)
    return MDI{IIP, F, S, P, D, T}(prob, u0, prob.t0, deepcopy(u0), prob.p)
end

function integrator(ds::DDS, u0::AbstractVector = ds.prob.u0)
    U0 = safe_state_type(ds, u0)
    if typeof(ds.prob) <: DiscreteProblem
        prob = DiscreteProblem(ds.prob.f, U0, DDS_TSPAN, ds.prob.p;
        callback = ds.prob.callback)
        integ = init(prob, FunctionMap(); save_everystep = false)
    elseif typeof(ds.prob) <: MDP
        integ = init(ds.prob, U0)
    else
        error("wtf")
    end
    return integ
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
