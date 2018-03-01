using DynamicalSystemsBase
using Base.Test, StaticArrays

println("\nTesting neighborhoods...")

@testset "Neighborhoods" begin
    ds = Systems.towel()
    data = trajectory(ds, 10000)

    tree = KDTree(data)

    point1 = data[100]
    point2 = point1 + 0.01rand(SVector{3})

    for ntype in [FixedMassNeighborhood(3), FixedSizeNeighborhood(0.1)]
        n1 = neighborhood(100, point1, tree, ntype)
        @test 100 ∉ n1
        n11 = neighborhood(point1, tree, ntype)
        @test 100 ∈ n11
    end
end
