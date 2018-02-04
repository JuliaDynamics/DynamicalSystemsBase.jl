using StaticArrays, ForwardDiff
import DiffEqBase: isinplace

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

mutable struct DiscreteLaw{IIP, D, T, S<:AbstractVector, F, P}
    s::S # s stands for state
    eom::F
    p::P
end

function DiscreteLaw(s::S, eom::F, p::P) where {S, F, P}
    D = length(s)
    T = eltype(s)
    iip = isinplace(eom, 3)
    DiscreteLaw{iip, D, T, S, F, P}(s, eom, p)
end

isinplace(::DiscreteLaw{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = IIP
dimension(::DiscreteLaw{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = D
state(dl::DiscreteLaw) = dl.s

struct DiscreteDS{IIP, D, T, S, F, P, J, M}
    system::DiscreteLaw{IIP, D, T, S, F, P}
    jacobian::J
    # The following 2 are used only in the case of IIP = true
    dummy::S
    J::M
end

function DiscreteDS(s::S, eom::F, p::P) where {S, F, P}
    system = DiscreteLaw(s, eom, p)
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
    return DiscreteDS(system, jacob, deepcopy(s), J)
end

isinplace(ds::DiscreteDS) = isinplace(ds.system)


# Test out of place:
p = [1.4, 0.3]
@inline henon_eom(x, p) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
ds = DiscreteDS(SVector(0.0, 0.0), henon_eom, p)
@assert !isinplace(ds)

# Test inplace:
function henon_eom_iip(dx, x, p)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
end

ds2 = DiscreteDS([0.0, 0.0], henon_eom_iip, p)
@assert isinplace(ds2)

println("success.")
