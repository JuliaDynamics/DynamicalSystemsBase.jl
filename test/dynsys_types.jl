if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays, OrdinaryDiffEq

# Test:
@inline @inbounds function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end
@inline @inbounds function loop(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end
@inline @inbounds function lorenz63_jacob(J, u, p, t)
    σ, ρ, β = p
    J[1,1] = -σ; J[1, 2] = σ
    J[2,1] = ρ - u[3]; J[2,3] = -u[1]
    J[3,1] = u[2]; J[3,2] = u[1]; J[3,3] = -β
    return nothing
end

@inline henon_eom(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
function henon_eom_iip(dx, x, p, n)
    @inbounds dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    @inbounds dx[2] = p[2]*x[1]
    return
end
@inbounds function henon_jacob_iip(J, x, p, n)
    J[1,1] = -2*p[1]*x[1]
    J[1,2] = 1.0
    J[2,1] = p[2]
    J[2,2] = 0.0
    return
end

u0 = rand(3)
p = [10, 28, 8/3]
ds1 = CDS(liip, u0, p)
ds2 = CDS(loop, u0, p)
@assert isinplace(ds1)
@assert !isinplace(ds2)

u0h = zeros(2)
ph = [1.4, 0.3]
he1 = DDS(henon_eom_iip, u0h, ph)
he2 = DDS(henon_eom, u0h, ph)
@assert isinplace(ds1)
@assert !isinplace(ds2)

J = rand(3,3)
