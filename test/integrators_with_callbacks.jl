using DiffEqCallbacks
using DynamicalSystemsBase
using OrdinaryDiffEq: Vern7, Tsit5
using LinearAlgebra

@testset "SavingCallback parallel" begin

kwargs = (abstol=1e-14, reltol=1e-14, solver=Vern7(), maxiters=1e9)
ds = Systems.lorenz()
d0 = 1e-9
T = 100.0

save_func(u, t, integrator) = LinearAlgebra.norm(u[1] - u[2])
saved_values = SavedValues(eltype(ds.t0), eltype(ds.u0[1]))
cb = SavingCallback(save_func, saved_values)

u0 = get_state(ds)
pinteg = parallel_integrator(ds, [u0, u0 + rand(SVector{3})*d0];
kwargs..., callback = cb)
step!(pinteg, T)
n = saved_values.saveval
t = saved_values.t
@test length(n) > 1000
# test that norm increases:
@test n[2] > n[1]
@test n[end] > n[5]
@test length(pinteg.sol.u) == 1
end


@testset "SavingCallback tangent" begin

kwargs = (abstol=1e-14, reltol=1e-14, solver=Tsit5())
ds = Systems.lorenz()
d0 = 1e-9
T = 100.0

save_func(u, t, integrator) = LinearAlgebra.norm(get_deviations(integrator)[1, :])
saved_values = SavedValues(eltype(ds.t0), eltype(ds.u0[1]))
cb = SavingCallback(save_func, saved_values)

u0 = get_state(ds)
pinteg = tangent_integrator(ds; kwargs..., callback = cb)
step!(pinteg, T)
n = saved_values.saveval
t = saved_values.t
@test length(n) > 1000
# test that norm increases:
@test mean(n[10000:11000]) > mean(n[1:100])
@test n[end] > n[5]
@test length(pinteg.sol.u) == 1

end

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
