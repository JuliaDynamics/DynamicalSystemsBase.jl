# if current_module() != DynamicalSystemsBase
#   using DynamicalSystemsBase
# end
using Base.Test, StaticArrays, OrdinaryDiffEq

orthonormal(D, k) = qr(rand(D, D))[1][:, 1:k]

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
@inline hoop(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
@inline hoop_jac(x, p, n) = @SMatrix [-2*p[1]*x[1] 1.0; p[2] 0.0]
function hiip(dx, x, p, n)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end
function hiip_jac(J, x, p, n)
    J[1,1] = -2*p[1]*x[1]
    J[1,2] = 1.0
    J[2,1] = p[2]
    J[2,2] = 0.0
    return
end

# minimalistic lyapunov
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
    λ = λ/tode.t # woooorks
end
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
    λ = λ/tode.t # woooorks
end


u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = zeros(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]

for i in 1:8
    @testset "combination $(FUNCTIONS[i])" begin
        sysindx = i < 5 ? 1 : 2
        if i < 5
            if isodd(i)
                ds = CDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = CDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        else
            if isodd(i)
                ds = DDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = DDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        end
        isad = isautodiff(ds)
        iip = isinplace(ds)
        @test isodd(i) ? isad : !isad
        @test mod(i-1, 4) < 2 ? iip : !iip
        J = jacobian(ds)
        @test typeof(J) <: (iip ? Matrix : SMatrix)
        # if iip
        #     λ = lyapunovs_iip(ds, 4-sysindx)
        # else
        #     λ = lyapunovs_oop(ds, 3-sysindx)
        # end

    end
end




#######################################################################################
#                                      TIMINGS                                        #
#######################################################################################
println("Starting timings (using @time)..")

println("\nAutodiff CDS (create/jacobian):")
println("IIP")
@time lip = CDS(liip, u0, p)
jacobian(lip); @time jacobian(lip)
println("OOP")
@time lop = CDS(loop, u0, p)
jacobian(lop); @time jacobian(lop)

println("\n CDS (create/jacobian):")
println("IIP")
@time lipjac = CDS(liip, u0, p, liip_jac)
jacobian(lipjac); @time jacobian(lipjac)
println("OOP")
@time lopjac = CDS(loop, u0, p, loop_jac)
jacobian(lopjac); @time jacobian(lopjac)

println("\nAutodiff DDS (create/jacobian):")
println("IIP")
@time hip = DDS(hiip, u0h, ph)
jacobian(hip); @time jacobian(hip)
println("OOP")
@time hop = DDS(hoop, u0h, ph)
jacobian(hop); @time jacobian(hop)

println("\nDDS (create/jacobian):")
println("IIP")
@time hipjac = DDS(hiip, u0h, ph, hiip_jac)
jacobian(hipjac); @time jacobian(hipjac)
println("OOP")
@time hopjac = DDS(hoop, u0h, ph, hoop_jac)
jacobian(hopjac); @time jacobian(hopjac)



### Stepping of integrators

println("\nTime to create & step! integrators (CDS OOP AUTODIFF)")
ds = lop
integ = integrator(ds)
@time integ = integrator(ds)
step!(integ)
@time step!(integ)
te = tangent_integrator(ds, orthonormal(3,3))
println("tangent")
@time te = tangent_integrator(ds, orthonormal(3,3))
# Compilation time of step! is quite absurd. (At around ~30 seconds on my laptop)
# I am guessing it is the time to compile all of the stepping process of
# OrdinaryDiffEq?
step!(te); step!(te)
@time step!(te)



println("\nTime to create & step! integrators (CDS OOP)")
ds = lopjac
integ = integrator(ds)
@time integ = integrator(ds)
step!(integ)
@time step!(integ)
te = tangent_integrator(ds, orthonormal(3,3))
println("tangent")
@time te = tangent_integrator(ds, orthonormal(3,3))
# Here the of compilation is muuuuch smaller
step!(te); step!(te)
@time step!(te)


println("\nTime to create & step! integrators (CDS IIP AUTODIFF)")
ds = lip
integ = integrator(ds)
@time integ = integrator(ds)
step!(integ)
@time step!(integ)
te = tangent_integrator(ds, orthonormal(3,3))
println("tangent")
@time te = tangent_integrator(ds, orthonormal(3,3))
step!(te); step!(te)
@time step!(te)

println("\nTime to create & step! integrators (CDS IIP)")
ds = lipjac
integ = integrator(ds)
@time integ = integrator(ds)
step!(integ)
@time step!(integ)
te = tangent_integrator(ds, orthonormal(3,3))
println("tangent")
@time te = tangent_integrator(ds, orthonormal(3,3))
step!(te); step!(te)
@time step!(te)








## Lyapunov Timings
println("TIME LYAPUNOV OOP")
lyapunov_oop(lopjac, 3)
@time lyapunov_oop(lopjac, 3)
