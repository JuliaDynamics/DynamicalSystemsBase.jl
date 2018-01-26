println("\nTesting discrete system evolution...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

@testset "Logistic Map" begin

  d1 = Systems.logistic(0.1)
  d2 = DiscreteDS1D(0.1, d1.eom; parameters = d1.p)
  d3 = DiscreteDS1D(big(0.1), d1.eom, d1.deriv;parameters = d1.p)

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
    f1 = d1.deriv(d1.state, d1.p)
    f2 = d2.deriv(d2.state, d2.p)
    f3 = d3.deriv(d3.state, d3.p)
    @test isapprox(f1, f2;rtol = 1e-12)
    @test isapprox(f1, f3;rtol = 1e-12)
    @test typeof(f3) == BigFloat
  end
end

@testset "DiscreteDS $v" for v in ["towel", "henon"]

  if v == "towel"
    N = 3
    s1 = Systems.towel(0.1ones(N))
  elseif v == "henon"
    N = 2
    s1 = Systems.henon(0.1ones(N))
  end

  s2 = DiscreteDS(0.1ones(N), s1.eom; parameters = s1.p)
  s4 = DiscreteDS(round.(big.(0.1ones(N)),3), s1.eom, s1.jacob; parameters = s1.p)

  @testset "Evolution & trajectory" begin
    st1 = evolve(s1, 1)
    st2 = evolve(s2, 1)
    st4 = evolve(s4, 1)

    @test isapprox.(st1, st2; rtol = 1e-12) == trues(s1.state)
    @test isapprox.(st1, st4; rtol = 1e-12) == trues(s1.state)

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

    Jbef = copy(ds.J)
    ds.jacob!(ds.J, evolve(ds, 1), ds.p)
    @test Jbef != ds.J
    ds.jacob!(Jbef, evolve(ds, 1), ds.p)
    @test Jbef == ds.J
end
