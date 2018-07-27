using DynamicalSystemsBase
using Test, StaticArrays, LinearAlgebra
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac

println("\nTesting dynamical systems...")
let
u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = ones(2)
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

        tinteg = tangent_integrator(ds, orthonormal(dimension(ds), dimension(ds)))
        tuprev = deepcopy(get_state(tinteg))
        step!(tinteg)
        @test tuprev != get_state(tinteg)

        integ = integrator(ds)
        uprev = deepcopy(get_state(integ))
        step!(integ)
        @test uprev != get_state(integ)

        if i < 5

            tt = tinteg.t
            while integ.t < tt
                step!(integ)
            end

            @test get_state(tinteg) ≈ integ(tt)
        else
            @test get_state(tinteg) == get_state(integ)
        end

        # Currently the in-place version does not work from DiffEq's side:
        if i > 2
            pinteg = parallel_integrator(ds, [INITCOD[sysindx], INITCOD[sysindx]])
            puprev = deepcopy(get_state(pinteg))
            step!(pinteg)
            @test get_state(pinteg, 1) == get_state(pinteg, 2) == get_state(pinteg)
            @test puprev != get_state(pinteg)

            if i < 5
                # The below code does not work at the moment because there
                # is no interpolation for Vector[SVector]

                # tt = pinteg.t
                # while integ.t < tt
                #     step!(integ)
                # end
                # @test state(pinteg)[1] ≈ integ(tt)

            else
                @test get_state(pinteg) == get_state(integ)
            end
        else
            pinteg = parallel_integrator(ds, [INITCOD[sysindx], INITCOD[sysindx]])
            puprev = deepcopy(get_state(pinteg))
            step!(pinteg)
            @test get_state(pinteg, 1) == get_state(pinteg, 2) == get_state(pinteg)
            @test puprev != get_state(pinteg)
            tt = pinteg.t
            while integ.t < tt
                step!(integ)
            end
            @test get_state(pinteg, 1) ≈ integ(tt)
        end
    end
end
end
