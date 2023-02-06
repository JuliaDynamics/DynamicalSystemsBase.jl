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

function projected_tests(ds, pds, P)
    @testset "projected dedicated" begin
    if P == 1
        y = current_state(pds)
        @test length(y) == 2
        @test dimension(pds) == 2
        reinit!(pds) # also reinits `ds`
        step!(pds) # also steps `ds`
        @test current_state(pds) == current_state(pds)[1:2]
    elseif P == 2
        y = current_state(pds)
        @test length(y) == 2
        @test dimension(pds) == 2
        reinit!(pds, [0.5, 0.5])
        @test current_state(ds) == [0.5, 0.5, 1.5]
    elseif P == 3
        @test norm(current_state(pds)) ≈ 1
        reinit!(pds, ones(3))
        @test current_state(ds)[1] ≈ 10
        @test current_state(ds)[3] ≈ 10
        step!(pds, 1)
        @test norm(current_state(pds)) ≈ 1
        @test current_time(ds) == current_time(pds) > 0
        @test dimension(pds) == 3
    end
    end
end

@testset "IDT=$(IDT), IIP=$(IIP) proj=$(P)" for IDT in (true, false), IIP in (false, true), P in (1, 2, 3)
    SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
    rule = !IIP ? trivial_rule : trivial_rule_iip
    p0 = IDT ? p0_disc : p0_cont
    ds = SystemType(rule, u0, p0)

    projection, complete = (proj_comp1, proj_comp2, proj_comp3)[P]
    pds = ProjectedDynamicalSystem(ds, projection, complete)
    u0init = recursivecopy(current_state(pds))

    if P ∈ (1, 2)
        test_dynamical_system(pds, u0init, p0;
        test_init_state_equiv=false, idt=IDT, iip=IIP)
    end

    # Specific poincare map tests here:
    projected_tests(ds, pds, P)
end
