using StaticArrays, ForwardDiff
import DiffEqBase: isinplace
import Base: eltype

export DiscreteDynamicalSystem, DDS, DiscreteProblem
export state, jacobian, isinplace, dimension, statetype, state

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

mutable struct DiscreteProblem{IIP, D, T, S<:AbstractVector, F, P}
    s::S # s stands for state
    eom::F
    p::P
end

function DiscreteProblem(s::S, eom::F, p::P) where {S, F, P}
    D = length(s)
    T = eltype(s)
    iip = isinplace(eom, 3)
    DiscreteProblem{iip, D, T, S, F, P}(s, eom, p)
end

isinplace(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = IIP
dimension(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = D
eltype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = T
statetype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = S
state(dl::DiscreteProblem) = dl.s

struct DiscreteDynamicalSystem{IIP, D, T, S, F, P, J, M}
    system::DiscreteProblem{IIP, D, T, S, F, P}
    jacobian::J
    # The following 2 are used only in the case of IIP = true
    dummy::S
    J::M
end

function DiscreteDynamicalSystem(s::S, eom::F, p::P) where {S, F, P}
    system = DiscreteProblem(s, eom, p)
    iip = isinplace(system)
    if !iip
        reducedeom = (x) -> eom(x, system.p)
    else
        reducedeom = (dx, x) -> eom(dx, x, system.p)
    end
    jacob = generate_jacobian(iip, reducedeom, s)
    J = begin
        D = dimension(system)
        if iip
            J = similar(s, (D,D))
            jacob(J, s, system.p)
        else
            J = jacob(s, system.p)
        end
    end
    return DiscreteDynamicalSystem(system, jacob, deepcopy(s), J)
end

for f in (:isinplace, :dimension, :eltype, :statetype, :state)
    @eval begin
        @inline ($f)(ds::DiscreteDynamicalSystem) = $(f)(ds.system)
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
end

ds2 = DDS([0.0, 0.0], henon_eom_iip, p)
@assert isinplace(ds2)


# set_state
function set_state!(ds::DDS, xnew)
    ds.system.s = xnew
end

xnew = rand(2)

set_state!(ds, xnew)
set_state!(ds2, xnew)

@assert state(ds) == xnew
@assert state(ds2) == xnew


function jacobian(ds::DDS{true, D, T, S, F, P, J, M},
    u = state(ds)) where {D, T, S, F, P, J, M}

    ds.jacobian(ds.J, u, ds.system.p)
    return ds.J
end

function jacobian(ds::DDS{false, D, T, S, F, P, J, M},
    u = state(ds)) where {D, T, S, F, P, J, M}
    ds.jacobian(u, ds.system.p)
end

@assert jacobian(ds) == jacobian(ds2)



println("success.")
