using DynamicalSystemsBase
using Test, StaticArrays
using DynamicalSystemsBase: DDS, DS

println("\nTesting discrete system evolution...")

@testset "Logistic Map" begin

  d1 = Systems.logistic(0.1)
  d2 = DDS(d1.f, 0.1, d1.p)
  d3 = DDS(d1.f, big(0.1), d1.p, d1.jacobian)

  @testset "trajectory" begin
    ts1 = trajectory(d1, 100)
    ts3 = trajectory(d3, 100)
    @test ts1[10] ≈ ts3[10]
    @test eltype(ts3) == BigFloat
  end
  @testset "Derivatives" begin
    f1 = jacobian(d1)
    f2 = jacobian(d2)
    f3 = jacobian(d3)
    @test isapprox(f1, f2;rtol = 1e-12)
    @test isapprox(f1, f3;rtol = 1e-12)
    @test typeof(f3) == BigFloat
  end
end

@testset "Henon map" begin

  d1 = Systems.henon()
  d2 = DDS(d1.f, get_state(d1), d1.p)
  d3 = DDS(d1.f, big.(get_state(d1)), d1.p, d1.jacobian)

  @testset "trajectory" begin
    ts1 = trajectory(d1, 100)
    ts3 = trajectory(d3, 100)
    @test ts1[10] ≈ ts3[10]
    @test eltype(ts3) == BigFloat
  end
  @testset "jacobian" begin
    f1 = jacobian(d1)
    f2 = jacobian(d2)
    f3 = jacobian(d3)
    @test isapprox(f1, f2;rtol = 1e-12)
    @test isapprox(f1, f3;rtol = 1e-12)
    @test eltype(f3) == BigFloat
  end
end

@testset "Coupled standard maps" begin
    M = 5; ks = 0.5ones(M); Γ = 0.05;
    ds = Systems.coupledstandardmaps(M, 0.1rand(2M); ks=ks, Γ = Γ)

    u0 = copy(get_state(ds))
    data = trajectory(ds, 100)
    @test u0 != data[end]

    @test jacobian(ds) != jacobian(ds, rand(2M))
end
