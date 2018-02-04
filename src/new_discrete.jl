using StaticArrays, ForwardDiff
import DiffEqBase: isinplace
import Base: eltype

export DiscreteDynamicalSystem, DDS, DiscreteProblem
export state, jacobian, isinplace, dimension, statetype, state

abstract type AbstractDynamicalSystem end

# Here f must be of the form: f(x) -> SVector (ONE ARGUMENT!)
function generate_jacobian_oop(f::F, x::X) where {F, X}
    # Test f structure:
    @assert !isinplace(f, 2)
    # Setup config
    cfg = ForwardDiff.JacobianConfig(f, x)
    FDjac(x, p) = ForwardDiff.jacobian(f, x, cfg)
    return FDjac
end

# Here f! must be of the form: f!(dx, x), in-place with 2 arguments!
function generate_jacobian_iip(f!::F, x::X) where {F, X}
    # Test f structure:
    @assert isinplace(f!, 2)
    # Setup config
    dum = deepcopy(x)
    cfg = ForwardDiff.JacobianConfig(f!, dum, x)
    # Notice that this version is inefficient: The result of applying f! is
    # already written in `dum` when the Jacobian is calculated. But this is
    # also done during normal evolution, making `f!` being applied twice.
    FDjac!(J, x, p) = ForwardDiff.jacobian!(J, f!, dum, x, cfg)
    return FDjac!
end

# At the moment this may be type-unstable, but on Julia 0.7 it will be stable
function generate_jacobian(iip::Bool, f::F, x::X) where {F, X}
    iip == true ? generate_jacobian_iip(f, x) : generate_jacobian_oop(f, x)
end

mutable struct DiscreteProblem{IIP, D, T, S<:AbstractVector{T}, F, P}
    s::S # s stands for state
    f::F # more similarity with ODEProblem
    p::P
end

function DiscreteProblem(s, eom::F, p::P) where {F, P}
    D = length(s)
    T = eltype(s)
    iip = isinplace(eom, 3)
    u = iip ? Vector(s) : SVector{D}(s)
    S = typeof(u)
    DiscreteProblem{iip, D, T, S, F, P}(u, eom, p)
end

isinplace(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = IIP
dimension(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = D
eltype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = T
statetype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = S
state(dl::DiscreteProblem) = dl.s

struct DiscreteDynamicalSystem{IIP, D, T, S, F, P, JA, M} <: AbstractDynamicalSystem
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    jacobian::JA
    # The following 2 are used only in the case of IIP = true
    dummy::S
    J::M
    # To solve DynamicalSystemsBase.jl#17
    isautodiff::Bool
end

function DiscreteDynamicalSystem(s::S, eom::F, p::P, jacob::JA) where {S, F, P, JA}
    prob = DiscreteProblem(s, eom, p)
    iip = isinplace(prob)
    J = begin
        D = dimension(prob)
        if iip
            J = similar(s, (D,D))
            jacob(J, s, prob.p)
        else
            J = jacob(s, prob.p)
        end
    end
    return DiscreteDynamicalSystem(prob, jacob, deepcopy(s), J, false)
end

function DiscreteDynamicalSystem(s::S, eom::F, p::P) where {S, F, P}
    prob = DiscreteProblem(s, eom, p)
    iip = isinplace(prob)
    if !iip
        reducedeom = (x) -> eom(x, prob.p)
    else
        reducedeom = (dx, x) -> eom(dx, x, prob.p)
    end
    jacob = generate_jacobian(iip, reducedeom, s)
    J = begin
        D = dimension(prob)
        if iip
            J = similar(s, (D,D))
            jacob(J, s, prob.p)
            J
        else
            J = jacob(s, prob.p)
        end
    end

    return DiscreteDynamicalSystem(prob, jacob, deepcopy(s), J, true)
end



for f in (:isinplace, :dimension, :eltype, :statetype, :state)
    @eval begin
        @inline ($f)(ds::DiscreteDynamicalSystem) = $(f)(ds.prob)
    end
end

# Alias
DDS = DiscreteDynamicalSystem

# Test out of place:
p = [1.4, 0.3]
@inline henon_eom(x, p) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
ds = DDS(SVector(0.0, 0.0), henon_eom, p)
@assert !isinplace(ds)

# Test inplace:
function henon_eom_iip(dx, x, p)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end

ds2 = DDS([0.0, 0.0], henon_eom_iip, p)
@assert isinplace(ds2)


# set_state
function set_state!(ds::DDS, xnew)
    ds.prob.s = xnew
end

xnew = rand(2)

set_state!(ds, xnew)
set_state!(ds2, xnew)

@assert state(ds) == xnew
@assert state(ds2) == xnew


function jacobian(ds::DDS{true}, u = state(ds))
    ds.jacobian(ds.J, u, ds.prob.p)
    return ds.J
end

jacobian(ds::DDS{false}, u = state(ds)) = ds.jacobian(u, ds.prob.p)

@assert jacobian(ds) == jacobian(ds2)


#= Methods necessary:
evolve(ds [, N] [, u])
evolve!([u], ds, N)
=#

evolve(ds::DDS{true}, u = state(ds)) = (ds.prob.f(ds.dummy, u, ds.prob.p); ds.dummy)
evolve(ds::DDS{false}, u = state(ds)) = ds.prob.f(u, ds.prob.p)

function evolve(ds::DDS{true}, N::Int, u = state(ds))
    D = dimension(ds)
    u0 = SVector{D}(u)
    ds.dummy .= u
    for i in 1:N
        ds.prob.f(ds.prob.s, ds.dummy, ds.prob.p)
        ds.dummy .= u
    end
    uret = SVector{D}(ds.prob.s)
    ds.prob.s .= u0
    return Vector(uret)
end

function evolve(ds::DDS{false}, N::Int, u0 = state(ds))
    for i in 1:N
        u0 = ds.prob.f(u0, ds.prob.p)
    end
    return u0
end

evolve!(u, ds::DDS{true}) = (ds.dummy .= u; ds.prob.f(u, ds.dummy, ds.prob.p))
function evolve!(u, ds::DDS{true}, N::Int)
    for i in 1:N
        ds.dummy .= u
        ds.prob.f(u, ds.dummy, ds.prob.p)
    end
    return
end
evolve!(ds::DDS{true}) = evolve!(ds.prob.s, ds)
evolve!(ds::DDS{true}, N::Int) = evolve!(ds.prob.s, ds, N)

evolve!(u, ds::DDS{false}) = (u .= ds.prob.f(u, ds.prob.p))
evolve!(u, ds::DDS{false}, N::Int) = (u .= evolve(ds, N, u))
evolve!(ds::DDS{false}, N::Int = 1) = (ds.prob.s = evolve(ds, N))



function trajectory(ds::DDS{true}, N::Int, u = state(ds))
    SV = SVector{dimension(ds), eltype(u)}
    f! = ds.prob.f
    ts = Vector{SV}(N)
    ts[1] = SV(u)
    for i in 2:N
        ds.dummy .= ts[i-1]
        f!(ds.prob.s, ds.dummy, ds.prob.p)
        ts[i] = SV(ds.prob.s)
    end
    ds.prob.s .= ts[1]
    return ts # Dataset(ts)
end

function trajectory(ds::DDS{false}, N::Int, st = state(ds))
    SV = SVector{dimension(ds), eltype(st)}
    ts = Vector{SV}(N)
    ts[1] = st
    f = ds.prob.f
    for i in 2:N
        st = f(st, ds.prob.p)
        ts[i] = st
    end
    return ts # Dataset(ts)
end

# Parallel evolver!
struct ParallelEvolver{IIP, D, T, S<:AbstractVector{T}, F, P, k}
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    states::Vector{S}
    # used only when IIP = true
    dummy::Vector{T}
end

function ParallelEvolver(prob::DiscreteProblem{IIP, D, T, S, F, P}, states) where
    {IIP, D, T, S<:AbstractVector{T}, F, P}
    k = length(states)
    if IIP == true
        s = [Vector(a) for a in states]
    else
        s = [SVector{D, T}(a) for a in states]
    end
    return ParallelEvolver{IIP, D, T, S, F, P, k}(prob, s, Vector(deepcopy(states[1])))
end

ParallelEvolver(ds::DDS, states) = ParallelEvolver(ds.prob, states)

function evolve!(pe::ParallelEvolver{true, D, T, S, F, P, k},
    N::Int = 1) where {D, T, S<:AbstractVector{T}, F, P, k}
    for j in 1:N
        for i in 1:k
            pe.dummy .= pe.states[i]
            pe.prob.f(pe.states[i], pe.dummy, pe.prob.p)
        end
    end
    return
end

function evolve!(pe::ParallelEvolver{false, D, T, S, F, P, k},
    N::Int = 1) where {D, T, S<:AbstractVector{T}, F, P, k}
    for j in 1:N
        for i in 1:k
            pe.states[i] = pe.prob.f(pe.states[i], pe.prob.p)
        end
    end
    return
end


states = [zeros(2), zeros(2)]

ds = DDS(SVector(0.0, 0.0), henon_eom, p)
ds2 = DDS([0.0, 0.0], henon_eom_iip, p)

pe = ParallelEvolver(ds.prob, states)
pe2 = ParallelEvolver(ds2.prob, states)


evolve!(pe, 10); evolve!(pe2, 10)
@assert pe.states[1] == pe2.states[1]
@assert pe.states[2] == pe2.states[2]
@assert pe.states[1] == pe.states[2]

@assert evolve(ds)

println("success.")

using BenchmarkTools
