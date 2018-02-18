using DynamicalSystemsBase
using Base.Test, StaticArrays
using DynamicalSystemsBase: CDS, DS
using Base.Test, StaticArrays, OrdinaryDiffEq
using DiffEqCallbacks

println("\nTesting continuous system evolution...")

# @testset "ODEProblem conservation" begin
#
#   lo11 = Systems.lorenz() #with Jac
#   lo22 = DS(lo11.prob) #without Jac
#
#   @test lo11.prob == lo22.prob
#
#   t = (0.0, 100.0)
#   p = [5,5,5.0]
#   # Event when event_f(t,u) == 0
#   condition = (u, t, integrator) -> u[1]
#
#   affect! = (integrator) -> (integrator.u[2] = -integrator.u[2])
#
#   cb = ContinuousCallback(condition, affect!)
#
#   prob1 = ODEProblem(lo11.prob.f, rand(SVector{3}), t, p; callback = cb)
#   ds = DS(prob1)
#   ds2 = DS(prob1, lo11.jacobian; J0 = lo11.J)
#
#   @test ds.prob.callback == cb
#   @test ds2.prob.callback == cb
#
# end


@testset "Lorenz System" begin

  lo11 = Systems.lorenz() #with Jac
  lo22 = CDS(lo11.prob.f, lo11.prob.u0, lo11.prob.p) #without Jac
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
    diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    ts3 = trajectory(lo33, 1.0; dt=0.1,
    diff_eq_kwargs=Dict(:abstol=>1e-8, :reltol=>1e-8))
    @test size(ts1) == size(ts3)
    @test ts1[end, :] ≈ ts3[end,:]
  end

end

@testset "Lorenz96" begin
    u = ones(5)
    lo11 = Systems.lorenz96(5, u)
    lo33 = Systems.lorenz96(5, big.(ones(5)))

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
  ds2 = CDS(ds1.prob) #without Jac
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
