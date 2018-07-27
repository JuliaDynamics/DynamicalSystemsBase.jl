using DynamicalSystemsBase
using Test, StaticArrays
using DynamicalSystemsBase: create_jacobian, create_tangent, stateeltype, isinplace

println("\nTesting inference...")

@testset "Inference" begin
    dss = [Systems.towel(), Systems.henon_iip()]
    @testset "IIP = $IIP" for IIP in [false, true]
        ds = IIP ? dss[2] : dss[1]
        @test_nowarn @inferred integrator(ds)

        f = ds.f
        s = get_state(ds)
        p = ds.p
        t = 0
        D = dimension(ds)

        @testset "Create Jac" begin
            # @test_nowarn @inferred create_jacobian(f, Val{IIP}(), s, p, t, Val{D}())
            @test_nowarn @inferred create_tangent(f, ds.jacobian, ds.J, Val{IIP}(), Val{2}())
            @test_nowarn @inferred jacobian(ds)
        end


        # Integrator state inference:
        @testset "Integrator" begin
            @test_nowarn @inferred stateeltype(ds)
            integ = integrator(ds)
            @test_nowarn @inferred stateeltype(integ)
            @test stateeltype(integ) == Float64
        end

        @testset "Tangent" begin
            integ = tangent_integrator(ds, 2)
            @test_nowarn @inferred stateeltype(integ)
            @test stateeltype(integ) == Float64
        end
        @testset "Parallel" begin
            integ = parallel_integrator(ds, [get_state(ds)])
            @test_nowarn @inferred stateeltype(integ)
            @test stateeltype(integ) == Float64
        end
    end
end
