using DynamicalSystemsBase, Test
using LinearAlgebra: norm
include("test_system_function.jl")

trivial_rule(x, p, n) = SVector{3}(p[1]*x[1], p[2]*x[2], p[3]*x[3])
function trivial_rule_iip(dx, x, p, n)
    dx .= trivial_rule(x, p, n)
    return
end

u0 = ones(3)
p0_disc = [1.1, 0.8, 0.9]
p0_cont = [0.1, -0.4, -0.2]
proj_comp1 = (1:2, [1.0])
proj_comp2 = (1:2, (y) -> [y[1], y[2], y[2] + 1])
proj_comp3 = (u -> u/norm(u), y -> 10y)

@testset "projected IDT=$(IDT), IIP=$(IIP) proj=$(P)" for IDT in (true, false), IIP in (false, true), P in (1, 2, 3)
    SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
    rule = !IIP ? trivial_rule : gissinger_rule_iip
    p0 = IDT ? p0_disc : p0_cont
    ds = SystemType(rule, u0, p0)

    projection, complete = (proj_comp1, proj_comp2, proj_comp3)[P]
    pds = ProjectedDynamicalSystem(ds, projection, complete)
    u0init = recursivecopy(current_state(pds))

    test_dynamical_system(pds, u0init, p0;
    test_init_state_equiv=false, idt=IDT, iip=IIP)
    # Specific poincare map tests here:
    # poincare_tests(ds, pmap, plane)
end


#=
@testset "Projected integrator" begin

@testset "Lorenz" begin
    # Set β to 10, we have a stable fixed point
    ds = Systems.lorenz([0.0, 10.0, 1.0]; σ = 10.0, ρ = 28.0, β = 10)
    psys = projected_integrator(ds, 1:2, [0.0])
    @test get_state(psys) == get_state(ds)[1:2]

    reinit!(psys, [1.0, 1.0])
    for j in 1:5
      step!(psys, 1.0)
    end
    y = get_state(psys)
    @test length(y) == 2
    @test abs(sum(y .- [16.43, 16.43])) < 0.01

    # Test complete state function
    complete = (y) -> [y[1], y[2], y[2] + 1]
    psys = projected_integrator(ds, 1:2, complete)
    reinit!(psys, [2.0, 1.0])
    y = get_state(psys)
    @test y[1] == 2
    @test length(y) == 2

    for j in 1:5
      step!(psys, 1.0)
    end
    u = get_state(psys)
    @test abs(sum(u - [16.43, 16.43])) < 0.01

    p0 = get_parameter(ds, 1)
    @test get_parameter(psys, 1) == p0
    set_parameter!(psys, 1, 0.5)
    @test get_parameter(psys, 1) == 0.5
    set_parameter!(psys, 1, p0)

    # Test projection function on a sphere of unit radius
    projection(u) = u/norm(u)
    complete = y -> 10y
    psys = projected_integrator(ds, projection, complete)
    @test norm(get_state(psys)) ≈ 1
    reinit!(psys, ones(3))
    @test psys.integ.u[1] == 10

    for j in 1:5
      step!(psys, 1.0)
    end
    u = get_state(psys)
    @test abs(sum(u[1:2] - [0.461, 0.461])) < 0.01

    # Trajectory, continuous
    psys = projected_integrator(ds, projection, complete)
    tr = trajectory(psys, 10)
    @test dimension(tr) == 3
    @test norm(tr[end]) ≈ 1

    # Trajectory, discrete
    ds = Systems.towel()
    psys = projected_integrator(ds, [1,2], [0.0])
    @test get_state(psys) == [0.085, -0.121]
    @test dimension(psys) == 2
    tr = trajectory(psys, 10)
    @test dimension(tr) == 2
end


end