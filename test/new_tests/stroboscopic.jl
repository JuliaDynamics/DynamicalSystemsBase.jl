using DynamicalSystemsBase, Test

using OrdinaryDiffEq: Vern9

include("test_system_function.jl")

@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
end
function duffing_rule_iip(du, u, p, t)
    du .= duffing_rule(u, p, t)
    return nothing
end

u0 = [0.1, 0.25]
p0 = [1.0, 0.3, 0.2, -1]
T = 2π/1.0

duffing_oop = StroboscopicMap(CoupledODEs(duffing_rule, u0, p0), T)
duffing_iip = StroboscopicMap(T, duffing_rule_iip, copy(u0), p0)
duffing_vern = StroboscopicMap(T, duffing_rule, u0, p0;
    diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)
)

for (ds, iip) in zip((duffing_oop, duffing_iip, duffing_vern), (false, true, false))
    name = (ds === duffing_vern) ? "duffvern" : "duffing"
    @testset "$name IIP=$(iip)" begin
        @test dynamic_rule(ds) == (iip ? duffing_rule_iip : duffing_rule)
        test_dynamical_system(ds, u0, p0; idt = true, iip)
    end
end

@testset "Duffing fixed point" begin
    pfp = (ω = 2.2, f = 27.0, d = 0.2, β = 1)
    duffing_fp = StroboscopicMap(CoupledODEs(duffing_rule, u0, pfp), 2π/2.2)
    X, t = trajectory(duffing_fp, 10; Ttr = 5000)
    @test X[end] ≈ X[end-1] atol = 1e-6
    @test X[end-1] ≈ X[end-2] atol = 1e-6
end

# @testset "chaotic attractor rotation" begin
#     # Animation of chaotic attractor in rotation
#     using CairoMakie
#     p0 = [1.0, 0.3, 0.2, -1]
#     T = 2π
#     ds = StroboscopicMap(CoupledODEs(duffing_rule, u0, p0), T)
#     t0s = range(0, 2T; length = 200)
#
#     fig = Figure()
#     ax = Axis(fig[1,1])
#     X, t = trajectory(ds, 1000; t0 = 0)
#     obs = Observable(vec(X))
#     scatter!(ax, obs)
#
#     record(fig, "duffing.mp4", t0s) do t0
#         X, t = trajectory(ds, 2000, current_state(ds); Ttr = 10, t0)
#         obs[] = vec(X)
#         # func(i) # or some other manipulation of the figure
#     end
#     @test true
# end
