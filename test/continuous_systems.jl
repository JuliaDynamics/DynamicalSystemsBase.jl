if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays, OrdinaryDiffEq

println("\nTesting continuous systems...")

@testset "ODEProblem conservation"

  lo11 = Systems.lorenz() #with Jac
  lo22 = ContinuousDS(lo11.prob) #without Jac

  @test lo11.prob == lo22.prob

  function condition(t,u,integrator) # Event when event_f(t,u) == 0
    u[1]
  end
  function affect!(integrator)
    integrator.u[2] = -integrator.u[2]
  end
  cb = ContinuousCallback(condition,affect!)



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
    j1 = (lo11.jacob!(0, s1, lo11.J); lo11.J)
    j2 = (lo22.jacob!(0, s2, lo22.J); lo22.J)
    j3 = (lo33.jacob!(0, s3, lo33.J); lo33.J)
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
