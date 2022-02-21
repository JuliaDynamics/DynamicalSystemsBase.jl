using DynamicalSystemsBase
using LinearAlgebra:norm
using Test

println("\nTesting continuous system evolution...")

@testset "Lorenz63 oop" begin

    lo11 = Systems.lorenz() #with Jac
    ts1 = trajectory(lo11, 1.0, Δt = 0.01)
    @test size(ts1) == (101, 3)
    @test ts1[1, :] == SVector{3}(lo11.u0)
    @test !any(x -> abs(x) > 1e3, ts1[end])

    ts1 = trajectory(lo11, 1.0; Δt=0.1, diffeq = (abstol=1e-8, reltol=1e-8))
    @test size(ts1) == (11, 3)
    @test ts1[1, :] == SVector{3}(lo11.u0)

    data2 = trajectory(lo11, 100; save_idxs = 1:2)
    @test size(data2)[2] == 2
end

@testset "Lorenz96 iin" begin
    u = ones(5)
    lo11 = Systems.lorenz96(5, u)
    ts1 = trajectory(lo11, 2.0, Δt = 0.01)
    @test size(ts1) == (201, 5)
    @test ts1[1, :] == SVector{5}(u)

    ts1 = trajectory(lo11, 2.0; Δt=0.1, diffeq = (abstol=1e-8, reltol=1e-8))
    @test size(ts1) == (21, 5)
    @test ts1[1, :] == SVector{5}(u)
    @test !any(x -> abs(x) > 1e3, ts1[end])
end

println("\nTesting stroboscopic map...")

@testset "Duffing stroboscopic map" begin
    F = 0.27; ω = 0.1;  # smooth boundary
    ds = Systems.duffing(zeros(2), ω = ω, f = F, d = 0.15, β = -1)
    smap = stroboscopicmap(ds, 2*pi/ω)
    reinit!(smap,[1., 1.])
    for j in 1:100
      step!(smap)
    end
    u = get_state(smap)
    @test abs(sum(u - [1.11, 0])) < 0.01
end
