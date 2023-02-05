using DynamicalSystemsBase, Test
using LinearAlgebra: cross

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

@testset "poincare IIP=$(IIP) plane=$(plane)" for IIP in (false, true), plane in (1, 2)
    ds = !IIP ? gissinger_oop : gissinger_iip
    plan = plane == 1 ? plane1 : plane2
    pmap = PoincareMap(ds, plan)
    u0 = initial_state(ds)
    test_dynamical_system(tands, u0, p; idt=true, iip=IIP, test_trajectory = false)
    # Specific poincare map tests here:
    # tangent_space_test(tands, lyapunovs)
end
