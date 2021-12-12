using DynamicalSystemsBase
using Test, StaticArrays, LinearAlgebra, OrdinaryDiffEq, SimpleDiffEq
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase: orthonormal
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac

println("\nTesting dynamical systems...")

algs = (Vern9(), Tsit5(), SimpleATsit5())

for alg in algs
u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = ones(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]

@testset "Dynamical system integrators" begin
for i in 1:8
    @testset "$alg combination $i" begin
        # Here we test the constructors with all possible cases of IIP/Autodiff
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

        diffeq = (alg = alg,)

        isad = DynamicalSystemsBase.isautodiff(ds)
        iip = DynamicalSystemsBase.isinplace(ds)
        @test isodd(i) ? isad : !isad
        @test mod(i-1, 4) < 2 ? iip : !iip
        J = jacobian(ds)
        @test typeof(J) <: (iip ? Matrix : SMatrix)

        tinteg = tangent_integrator(ds, orthonormal(dimension(ds), dimension(ds)); alg = alg)
        tuprev = deepcopy(get_state(tinteg))
        step!(tinteg)
        @test tuprev != get_state(tinteg)

        integ = integrator(ds; diffeq)
        uprev = deepcopy(get_state(integ))
        step!(integ)
        @test uprev != get_state(integ)

        # Test that progressing tangent integrator gives same state as normal integrator
        if i < 5
            tt = tinteg.t
            while integ.t < tt
                step!(integ)
            end
            @test get_state(tinteg) ≈ integ(tt) atol = 1e-2
        else
            @test get_state(tinteg) == get_state(integ)
        end

        # Test parallel integrators
        pinteg = parallel_integrator(ds, [copy(INITCOD[sysindx]), copy(INITCOD[sysindx])]; alg = alg)
        puprev = deepcopy(get_state(pinteg))
        step!(pinteg)
        @test get_state(pinteg, 1) == get_state(pinteg, 2) == get_state(pinteg)
        @test puprev != get_state(pinteg)
        if i ∈ (1,2) 
            # Interpolation does not work for Vector{SVector} so it is tested for matrix
            tt = pinteg.t
            while integ.t < tt
                step!(integ)
            end
            @test get_state(pinteg, 1) ≈ integ(tt) atol = 1e-2
        elseif i > 4
            @test get_state(pinteg) == get_state(integ)
        end
        u2 = 2get_state(pinteg, 2)
        set_state!(pinteg, u2, 2)
        @test get_state(pinteg, 2) == u2
    end
end

end
end