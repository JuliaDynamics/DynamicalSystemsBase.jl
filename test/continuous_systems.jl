using DynamicalSystemsBase
using DynamicalSystemsBase: CDS, DS
using Test, StaticArrays, OrdinaryDiffEq

println("\nTesting continuous system evolution...")

@testset "Lorenz System" begin

  lo11 = Systems.lorenz() #with Jac
  # lo33 = Systems.lorenz(big.([0.0, 10.0, 0.0]))

  @testset "trajectory" begin
    # trajectory pure:
    ts1 = trajectory(lo11, 1.0, dt = 0.01)
    # ts3 = trajectory(lo33, 1.0)
    # @test eltype(ts3[1]) == BigFloat
    @test size(ts1) == (101, 3)
    # @test ts1[1, :] ≈ ts3[end,:]
    @test ts1[1, :] == SVector{3}(lo11.u0)
    # trajectory with diff_eq_kwargs and dt:
    ts1 = trajectory(lo11, 1.0; dt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    # ts3 = trajectory(lo33, 1.0; dt=0.1,
    # diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    @test size(ts1) == (11, 3)
    @test ts1[1, :] == SVector{3}(lo11.u0)
    # @test ts1[end, :] ≈ ts3[end,:]
  end

end

@testset "Lorenz96" begin
    u = ones(5)
    lo11 = Systems.lorenz96(5, u)
    # lo33 = Systems.lorenz96(5, big.(ones(5)))

    @testset "trajectory" begin
      # trajectory pure:
      ts1 = trajectory(lo11, 2.0, dt = 0.01)
      # ts3 = trajectory(lo33, 2.0)
      # @test eltype(ts3[1]) == BigFloat
      @test size(ts1) == (201, 5)
      # @test ts1[end, :] ≈ ts3[end,:]
      @test ts1[1, :] == SVector{5}(u)
      # trajectory with diff_eq_kwargs and dt:
      ts1 = trajectory(lo11, 2.0; dt=0.1,
      diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      # ts3 = trajectory(lo33, 2.0; dt=0.1,
      # diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      @test size(ts1) == (21, 5)
      # @test ts1[end, :] ≈ ts3[end,:]
      @test ts1[1, :] == SVector{5}(u)
    end
end

@testset "Gissinger Columns" begin
    ds = Systems.gissinger()
    data = trajectory(ds, 100.0)

    xyz = columns(data)
    x, y, z = columns(data)

    for i in 1:3
        @test xyz[i] == data[:, i]
    end
end
