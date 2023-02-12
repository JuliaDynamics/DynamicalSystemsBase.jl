using DynamicalSystemsBase, Test
using LinearAlgebra: cross, norm, dot
include("test_system_function.jl")

function gissinger_rule(u, p, t)
    μ, ν, Γ = p
    du1 = μ*u[1] - u[2]*u[3]
    du2 = -ν*u[2] + u[1]*u[3]
    du3 = Γ - u[3] + u[1]*u[2]
    return SVector{3}(du1, du2, du3)
end
gissinger_rule_iip(du, u, p, t) = (du .= gissinger_rule(u, p, t); nothing)

u0 = [
    -1.965640649646121
    0.40575505636444326
    0.04524022744226266
]
μ = 0.119
ν = 0.1
Γ = 0.9
p = [μ, ν, Γ]

gissinger_oop = CoupledODEs(gissinger_rule, u0, p)
gissinger_iip = CoupledODEs(gissinger_rule_iip, recursivecopy(u0), p)

# Define appropriate hyperplane for gissinger system
plane1 = (1, 0.0)
# I want hyperperplane defined by these two points:
Np(μ) = SVector{3}(sqrt(ν + Γ*sqrt(ν/μ)), -sqrt(μ + Γ*sqrt(μ/ν)), -sqrt(μ*ν))
Nm(μ) = SVector{3}(-sqrt(ν + Γ*sqrt(ν/μ)), sqrt(μ + Γ*sqrt(μ/ν)), -sqrt(μ*ν))
# Create hyperplane using normal vector to vector connecting points:
gis_plane(μ) = [cross(Np(μ), Nm(μ))..., 0]
plane2 = gis_plane(μ)

function poincare_tests(ds, pmap, plane)
    P, t = trajectory(pmap, 10)
    reinit!(ds)
    P2 = poincaresos(ds, plane, 100)
    @test length(P) == 11
    @test P[1] == P2[1]
    @test length(P2) > 1
    if plane isa Tuple # test that 0 is first element approximately
        @test all(x -> abs(x) < 1e-6, P[:, 1])
        @test all(x -> abs(x) < 1e-6, P2[:, 1])
    else
        @test all(x -> abs(x) > 1e-6, P[:, 1])
        @test all(x -> abs(x) > 1e-6, P2[:, 1])
        # Here we access internal field to get distance from plane
        @test all(u -> abs(pmap.planecrossing(u)) < 1e-6, P)
        @test all(u -> abs(pmap.planecrossing(u)) < 1e-6, P2)
    end
end

@testset "IIP=$(IIP) plane=$(P)" for IIP in (false, true), P in (1, 2)
    rule = !IIP ? gissinger_rule : gissinger_rule_iip
    ds = CoupledODEs(rule, recursivecopy(u0), p)
    plane = P == 1 ? plane1 : plane2
    pmap = PoincareMap(ds, plane)
    u0pmap = recursivecopy(current_state(pmap))
    test_dynamical_system(pmap, u0pmap, p;
    idt=true, iip=IIP, test_trajectory = true,
    test_init_state_equiv=false, u0init = initial_state(pmap))
    # Specific poincare map tests here:
    poincare_tests(ds, pmap, plane)
end

@testset "set_parameter works" begin
    function lorenz_rule(u, p, t)
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
    u0 = fill(10.0, 3)
    p = [10, 28, 8/3]
    plane = (1, 0.0)
    ds = CoupledODEs(lorenz_rule, recursivecopy(u0), p)

    pmap = PoincareMap(ds, plane)
    P, t = trajectory(pmap, 10)
    @test all(x -> abs(x) < 1e-6, P[:, 1])

    set_parameter!(pmap, 2, 26.4)

    P, t = trajectory(pmap, 10, nothing)
    @test all(x -> abs(x) < 1e-6, P[:, 1])

end