println("\nTesting discrete system evolution...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

@testset "Logistic Map" begin

  d1 = Systems.logistic(0.1)
  d2 = DDS(0.1, d1.prob.f, d1.prob.p)
  d3 = DDS(big(0.1), d1.prob.f, d1.prob.p, d1.jacobian)

  @testset "Evolution & trajectory" begin
    st1 = evolve(d1, 1)
    st2 = evolve(d2, 1)
    st3 = evolve(d3, 1)
    @test st1 == st2
    @test st1 ≈ st3
    @test typeof(st3) == BigFloat
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

@testset "DDS $v" for v in ["towel", "henon"]

  if v == "towel"
    N = 3
    s1 = Systems.towel(0.1ones(N))
  elseif v == "henon"
    N = 2
    s1 = Systems.henon(0.1ones(N))
  end

  s2 = DDS(0.1ones(N), s1.prob.f, s1.prob.p)
  s4 = DDS(round.(big.(0.1ones(N)),3), s1.prob.f, s1.prob.p, s1.jacobian)

  @testset "Evolution & trajectory" begin
    st1 = evolve(s1, 1)
    st2 = evolve(s2, 1)
    st4 = evolve(s4, 1)

    @test isapprox.(st1, st2; rtol = 1e-12) == trues(state(s1))
    @test isapprox.(st1, st4; rtol = 1e-12) == trues(state(s1))

    ts = trajectory(s1, 100)
    @test size(ts) == (100, N)
    ts4 = trajectory(s4, 100)
    @test size(ts4) == (100, N)
    @test eltype(ts4) == BigFloat
    @test isapprox.(ts[10, :],ts4[10, :]) == trues(N)
  end
  @testset "Jacobians" begin

    J1 = jacobian(s1)
    @test typeof(J1) <: SMatrix
    J2 = jacobian(s2)
    J4 = jacobian(s4)
    @test typeof(J4) <: SMatrix

    @test isapprox.(J1, J2; rtol = 1e-6) == trues(J1)
    @test isapprox.(J1, J4; rtol = 1e-6) == trues(J1)
    @test eltype(J4) == BigFloat
  end
end

@testset "Coupled standard maps" begin
    M = 5; ks = 0.5ones(M); Γ = 0.05;
    ds = Systems.coupledstandardmaps(M, 0.1rand(2M); ks=ks, Γ = Γ)

    u0 = copy(state(ds))
    st1 = evolve(ds, 100)

    @test st1 != u0
    @test u0 == state(ds)

    Jbef = deepcopy(ds.J)
    ds.jacobian(ds.J, evolve(ds, 1), ds.prob.p)
    @test Jbef != ds.J
    ds.jacobian(Jbef, evolve(ds, 1), ds.prob.p)
    @test Jbef == ds.J
end
