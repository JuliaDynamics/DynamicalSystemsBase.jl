using DiffEqCallbacks

#=
@testset "ManifoldProjection" begin
  ds1 = Systems.henonheiles() #with Jac
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
