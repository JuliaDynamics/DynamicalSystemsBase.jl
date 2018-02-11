# if current_module() != DynamicalSystemsBase
#   using DynamicalSystemsBase
# end
using Base.Test, StaticArrays, OrdinaryDiffEq

## Lorenz
@inline @inbounds function liip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end
@inline @inbounds function liip_jac(J, u, p, t)
    σ, ρ, β = p
    J[1,1] = -σ; J[1, 2] = σ; J[1,3] = 0
    J[2,1] = ρ - u[3]; J[2,2] = -1; J[2,3] = -u[1]
    J[3,1] = u[2]; J[3,2] = u[1]; J[3,3] = -β
    return nothing
end
@inline @inbounds function loop(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end
@inline @inbounds function loop_jac(u, p, t)
    σ, ρ, β = p
    J = @SMatrix [-σ  σ  0;
    ρ - u[3]  (-1)  (-u[1]);
    u[2]   u[1]  -β]
    return J
end

# Henon
@inline henon_eom(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
@inline henon_jacob(x, p) = @SMatrix [-2*p[1]*x[1] 1.0; p[2] 0.0]
function henon_eom_iip(dx, x, p, n)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end
function henon_jacob_iip(J, x, p, n)
    J[1,1] = -2*p[1]*x[1]
    J[1,2] = 1.0
    J[2,1] = p[2]
    J[2,2] = 0.0
    return
end


u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
ds1 = CDS(liip, u0, p)
ds2 = CDS(loop, u0, p)
@assert isinplace(ds1)
@assert !isinplace(ds2)

u0h = zeros(2)
ph = [1.4, 0.3]
he1 = DDS(henon_eom_iip, u0h, ph)
he2 = DDS(henon_eom, u0h, ph)

function lyapunov_iip(ds::DS, k)
    D = dimension(ds)
    tode = tangent_integrator(ds, orthonormal(D,k))
    λ = zeros(D)
    for t in 1:1000
        while tode.t < t
            step!(tode)
        end
        # println("K = $K")
        Q, R = qr(view(tode.u, :, 2:D+1))
        λ .+= log.(abs.(diag(R)))

        view(tode.u, :, 2:D+1) .= Q
        u_modified!(tode, true)
    end
    λ = λ/1000.0 # woooorks
end

# inplace, continuous, autodiff
lyapunov_iip(ds1, 3)

# inplace, discrete, autodiff
lyapunov_iip(he1, 2)

# inplace, discrete, userjac
he1jac = DDS(henon_eom_iip, u0h, ph, henon_jacob_iip)
lyapunov_iip(he1jac, 2)

# inplace, continuous, userjac
lipjac = CDS(liip, u0, p, liip_jac)
lyapunov_iip(lipjac, 3)




function lyapunov_oop(ds::DS, k)
    D = dimension(ds)
    tode = tangent_integrator(ds, orthonormal(D,k))
    λ = zeros(D)
    ws_idx = SVector{k, Int}(collect(2:k+1))
    for t in 1:1000
        while tode.t < t
            step!(tode)
        end
        # println("K = $K")
        Q, R = qr(tode.u[:, ws_idx])
        λ .+= log.(abs.(diag(R)))

        tode.u = hcat(tode.u[:,1], Q)
        u_modified!(tode, true)
    end
    λ = λ/1000.0 # woooorks
end

# out-of-place, continuous, userjac
ds = lopjac = CDS(loop, u0, p, loop_jac)
lyapunov_oop(lopjac, 3)
