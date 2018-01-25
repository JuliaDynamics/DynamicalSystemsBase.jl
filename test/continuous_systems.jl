if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays, OrdinaryDiffEq
# using DiffEqCallbacks

println("\nTesting continuous systems...")

#= Commenting out until Callbacks are on v3
@testset "ODEProblem conservation" begin

  lo11 = Systems.lorenz() #with Jac
  lo22 = ContinuousDS(lo11.prob) #without Jac

  @test lo11.prob == lo22.prob

  t = (0.0, 100.0)

  # Event when event_f(t,u) == 0
   condition(u, t, integrator) = u[1]

  function affect!(integrator)
    integrator.u[2] = -integrator.u[2]
  end

  cb = ContinuousCallback(condition, affect!)

  prob1 = ODEProblem(lo11.prob.f, rand(3), t, callback = cb)
  ds = ContinuousDS(prob1)
  ds2 = ContinuousDS(prob1, lo11.jacob!, lo11.J)

  @test ds.prob.callback == cb
  @test ds2.prob.callback == cb
  @test ds2.jacob! == lo11.jacob!

end
=#

@testset "Lorenz System" begin

  lo11 = Systems.lorenz() #with Jac
  lo22 = ContinuousDS(lo11.prob) #without Jac
  lo33 = Systems.lorenz(big.([0.0, 10.0, 0.0]))

  @testset "Evolve & kwargs" begin
    # test evolve(system):
    st1 = evolve(lo11, 1.0)
    st3 = evolve(lo33, 1.0)
    @test st1 ≈ st3
    # Test evolve(system,keywords):
    st1 = evolve(lo11, 1.0;
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    st3 = evolve(lo33, 1.0;
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    @test st1 ≈ st3

    evolve!(lo22, 1.0;
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    @test lo22.prob.u0 ≈ st1
    lo22.prob.u0 .= lo11.prob.u0
  end

  lo11 = Systems.lorenz() #with Jac
  lo22 = ContinuousDS(lo11.prob) #without Jac
  lo33 = Systems.lorenz(big.([0.0, 10.0, 0.0]))

  @testset "trajectory" begin
    # trajectory pure:
    ts1 = trajectory(lo11, 1.0)
    ts3 = trajectory(lo33, 1.0)
    @test eltype(ts3[1]) == BigFloat
    @test size(ts1) == size(ts3)
    @test ts1[end, :] ≈ ts3[end,:]
    # trajectory with diff_eq_kwargs and dt:
    ts1 = trajectory(lo11, 1.0; dt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    ts3 = trajectory(lo33, 1.0; dt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
    @test size(ts1) == size(ts3)
    @test ts1[end, :] ≈ ts3[end,:]
  end

  @testset "Jacobians" begin
    j1 = lo11.J
    j2 = lo22.J
    j3 = lo33.J
    @test eltype(j3) == BigFloat
    @test j1 ≈ j2
    @test j1 ≈ j3
    s1 = evolve(lo11, 1.0)
    s2 = evolve(lo22, 1.0)
    s3 = evolve(lo33, 1.0)
    lo11.jacob!(j1, s1, lo11.prob.p, 0)
    lo11.jacob!(j2, s2, lo11.prob.p, 0)
    lo33.jacob!(j3, s3, lo11.prob.p, 0)
    @test eltype(j3) == BigFloat
    @test j1 ≈ j2
    @test j1 ≈ j3
  end

end

@testset "Lorenz96" begin
    u = ones(5)
    lo11 = Systems.lorenz96(5, u)
    lo33 = Systems.lorenz96(5, big.(ones(5)))

    @testset "Evolve & kwargs" begin
      # test evolve(system):
      st1 = evolve(lo11, 1.0)
      st3 = evolve(lo33, 1.0)
      @test st1 ≈ st3
      # Test evolve(system,keywords):
      st1 = evolve(lo11, 1.0;
      diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      st3 = evolve(lo33, 1.0;
      diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      @test st1 ≈ st3
    end

    @testset "trajectory" begin
      # trajectory pure:
      ts1 = trajectory(lo11, 2.0)
      ts3 = trajectory(lo33, 2.0)
      @test eltype(ts3[1]) == BigFloat
      @test size(ts1) == size(ts3)
      @test ts1[end, :] ≈ ts3[end,:]
      # trajectory with diff_eq_kwargs and dt:
      ts1 = trajectory(lo11, 2.0; dt=0.1,
      diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      ts3 = trajectory(lo33, 2.0; dt=0.1,
      diff_eq_kwargs=Dict(:abstol=>1e-9, :reltol=>1e-9))
      @test size(ts1) == size(ts3)
      @test ts1[end, :] ≈ ts3[end,:]
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

#=
@testset "ManifoldProjection" begin
  ds1 = Systems.henonhelies() #with Jac
  ds2 = ContinuousDS(ds1.prob) #without Jac
  @inline Vhh(q1, q2) = 1//2 * (q1^2 + q2^2 + 2q1^2 * q2 - 2//3 * q2^3)
  @inline Thh(p1, p2) = 1//2 * (p1^2 + p2^2)
  @inline Hhh(q1, q2, p1, p2) = Thh(p1, p2) + Vhh(q1, q2)
  @inline Hhh(u::AbstractVector) = Hhh(u...)

  tra1 = trajectory(ds1, 100.0)
  tra2 = trajectory(ds2, 100.0)

  E1 = [Hhh(p) for p in tra1]
  E2 = [Hhh(p) for p in tra2]

  @test std(E1) < 1e-12
  @test std(E2) < 1e-12
end
=#
