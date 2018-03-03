using DynamicalSystemsBase
using Base.Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS, isinplace
using DynamicalSystemsBase: create_jacobian, create_tangent

println("\nTesting inference...")

@testset "Inference" begin
    for ds in [Systems.towel(), Systems.henon_iip()]
        @test_nowarn @inferred integrator(ds)
        f = ds.prob.f
        s = state(ds)
        p = ds.prob.p
        t = 0
        D = dimension(ds)
        IIP = isinplace(ds)
        @test_nowarn @inferred create_jacobian(f, Val{IIP}(), s, p, t, Val{D}())
        @test_nowarn @inferred create_tangent(f, ds.jacobian, ds.J, Val{IIP}(), Val{2}())
        @test_nowarn @inferred jacobian(ds)

        # Integrator state inference:
        @test_nowarn @inferred stateeltype(ds)
        integ = integrator(ds)
        @test_nowarn @inferred stateeltype(integ)
        @test stateeltype(integ) == Float64

        integ = tangent_integrator(ds, 2)
        @test_nowarn @inferred stateeltype(integ)
        @test stateeltype(integ) == Float64

        integ = parallel_integrator(ds, [state(ds)])
        @test_nowarn @inferred stateeltype(integ)
        @test stateeltype(integ) == Float64
    end
end
