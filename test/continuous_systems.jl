using DynamicalSystemsBase
using DynamicalSystemsBase: CDS, DS, columns
using Test, StaticArrays, OrdinaryDiffEq

println("\nTesting continuous system evolution...")

@testset "Lorenz63 oop" begin

    lo11 = Systems.lorenz() #with Jac
    # lo33 = Systems.lorenz(big.([0.0, 10.0, 0.0]))
    # trajectory pure:
    ts1 = trajectory(lo11, 1.0, Δt = 0.01)
    # ts3 = trajectory(lo33, 1.0)
    # @test eltype(ts3[1]) == BigFloat
    @test size(ts1) == (101, 3)
    # @test ts1[1, :] ≈ ts3[end,:]
    @test ts1[1, :] == SVector{3}(lo11.u0)
    @test !any(x -> abs(x) > 1e3, ts1[end])
    # trajectory with diff_eq_kwargs and Δt:
    ts1 = trajectory(lo11, 1.0; Δt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    # ts3 = trajectory(lo33, 1.0; Δt=0.1,
    # diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    @test size(ts1) == (11, 3)
    @test ts1[1, :] == SVector{3}(lo11.u0)
    # @test ts1[end, :] ≈ ts3[end,:]

    data2 = trajectory(lo11, 100; save_idxs = 1:2)
    @test size(data2)[2] == 2
end

@testset "Lorenz96 iin" begin
    u = ones(5)
    lo11 = Systems.lorenz96(5, u)
    # lo33 = Systems.lorenz96(5, big.(ones(5)))
    # trajectory pure:
    ts1 = trajectory(lo11, 2.0, Δt = 0.01)
    # ts3 = trajectory(lo33, 2.0)
    # @test eltype(ts3[1]) == BigFloat
    @test size(ts1) == (201, 5)
    # @test ts1[end, :] ≈ ts3[end,:]
    @test ts1[1, :] == SVector{5}(u)
    # trajectory with diff_eq_kwargs and Δt:
    ts1 = trajectory(lo11, 2.0; Δt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    # ts3 = trajectory(lo33, 2.0; Δt=0.1,
    # diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    @test size(ts1) == (21, 5)
    # @test ts1[end, :] ≈ ts3[end,:]
    @test ts1[1, :] == SVector{5}(u)
    @test !any(x -> abs(x) > 1e3, ts1[end])
end
