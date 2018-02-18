using DynamicalSystemsBase
using Base.Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS

println("\nTesting dynamical systems...")

# test inference of create_jacobian for both systems
# test inference of tangentf for both systems
# test inference of integrator for discrete

@testset "Discrete Inference" begin
    for ds in [Systems.towel(), Systems.henon_iip()]
        @inferred integrator(ds)
        f = ds.prob.f
        s = state(ds)
        p = ds.prob.p
        t = 0
        D = dimension(ds)
        IIP = isinplace(ds)
        @inferred create_jacobian(f, Val{IIP}(), s, p, t, Val{D}())
        @inferred create_tangent(f, ds.jacobian, ds.J, Val{IIP}(), Val{2}())
    end
end
