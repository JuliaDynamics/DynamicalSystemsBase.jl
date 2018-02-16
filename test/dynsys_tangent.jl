using DynamicalSystemsBase
using Base.Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac

u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = zeros(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]

for i in 1:8
    @testset "combination $i" begin
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

        isad = DynamicalSystemsBase.isautodiff(ds)
        iip = DynamicalSystemsBase.isinplace(ds)
        @test isodd(i) ? isad : !isad
        @test mod(i-1, 4) < 2 ? iip : !iip
        J = jacobian(ds)
        @test typeof(J) <: (iip ? Matrix : SMatrix)

        for ff in [integrator, tangent_integrator, parallel_integrator]
            integ = ff(ds)
            uprev = state(integ)
            step!(integ)
            @test uprev != state(integ)
        end

        # if iip
        #     λ = lyapunovs_iip(ds, 4-sysindx)
        # else
        #     λ = lyapunovs_oop(ds, 3-sysindx)
        # end

    end
end



error()

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
    λ./tode.t # woooorks
end




## Lyapunov Timings
println("TIME LYAPUNOV OOP")
lyapunov_oop(lopjac, 3)
@time lyapunov_oop(lopjac, 3)
lyapunov_oop(hopjac, 2)
@time lyapunov_oop(hopjac, 2)
# There seems to be massive slowdown on lyapunov of discrete. I expected 2
# orders of magnitude speed up. Plus old chaos tools found the exponents in
# 1 milisecond if I recall (for discerete). Maybe there is something
# wrong here.
