using DynamicalSystemsBase
using Test, StaticArrays

println("\nTesting neighborhoods...")

@testset "Neighborhoods" begin

    ds = Systems.towel()
    data = trajectory(ds, 10000)

    tree = KDTree(data)

    @testset "neighborhood" begin

        point1 = data[100]
        point2 = point1 + 0.000001one(SVector{3})

        kn1 = [1724, 100, 6765]

        ntype = FixedMassNeighborhood(3)
        n1 = neighborhood(point1, tree, ntype)
        @test n1 == kn1
        n11 = neighborhood(point2, tree, ntype)
        @test n11 == kn1

        kn1 = [100, 6765]
        ntype = FixedSizeNeighborhood(0.01)
        n1 = neighborhood(point1, tree, ntype)
        @test n1 == kn1
        n11 = neighborhood(point2, tree, ntype)
        @test n11 == kn1
    end

    @testset "Theiler" begin

        for ntype in [FixedMassNeighborhood(3), FixedSizeNeighborhood(0.1)]
            n1 = neighborhood(point1, tree, ntype, 100, 1)
            @test 100 ∉ n1
            n11 = neighborhood(point2, tree, ntype, 100, 0)
            @test 100 ∈ n11
        end

        ds = Systems.lorenz()
        data = trajectory(ds, 200, dt = 0.001)
        tree = KDTree(data)

        point1 = data[10000]

        for ntype in [FixedMassNeighborhood(3), FixedSizeNeighborhood(0.1)]

            n1 = neighborhood(point1, tree, ntype)

            typeof(ntype) <: FixedMassNeighborhood &&
                @test sort(n1) == [9999, 10000, 10001]

            n2 = neighborhood(point1, tree, ntype, 10000, 10)
            for i in 1:3
                @test abs(10000 - i) ≥ 10
            end
        end
    end
end
