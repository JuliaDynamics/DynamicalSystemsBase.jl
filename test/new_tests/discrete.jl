using DynamicalSystemsBase, Test

# Creation of Henon map
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
function henon_rule_iip(dx, x, p, n)
    dx .= henon_rule(x, p, n)
    return
end

u0 = zeros(2)
p0 = [1.4, 0.3]

henon_oop = DeterministicIteratedMap(henon_rule, u0, p0)
henon_iip = DeterministicIteratedMap(henon_rule_iip, u0, p0)

@testset "discr. map $(b)" for (henon, b) in zip((henon_oop, henon_iip), ("oop", "iip"))

@testset "obtaining info" begin
    @test current_state(henon) == u0
    @test initial_state(henon) == u0
    @test current_parameters(henon) == p0
    @test initial_parameters(henon) == p0
    @test current_time(henon) == 0
    @test initial_time(henon) == 0
    @test isinplace(henon) == (b == "iip")
    @test isdeterministic(henon) == true
    @test isdiscretetime(henon) == true
    @test dynamic_rule(henon) == (b == "oop" ? henon_rule : henon_rule_iip)
    @test henon(0) == u0
    @test_throws ArgumentError henon(2)
end

@testset "time evolution" begin
    step!(henon)
    @test current_time(henon) == 1
    @test current_state(henon) == [1, 0]
    @test henon(1) == [1, 0]
    step!(henon, 2)
    @test current_time(henon) == 3
    @test current_state(henon) != [1, 0] != u0
    step!(henon, 1, true)
    @test current_time(henon) == 4
end

@testset "alteration" begin
    set_parameter!(henon, 1, 1.0)
    @test current_parameters(henon)[1] == 1
    set_parameters!(henon, [2.0])
    @test current_parameters(henon)[1] == 2.0
    set_parameters!(henon, [2.0, 0.1])
    @test current_parameters(henon)[2] == 0.1

    reinit!(henon; p0 = initial_parameters(henon))
    @test henon(0) == u0
    @test current_state(henon) == u0
    @test current_parameters(henon)[1] == 1.4
end

@testset "trajectory" begin

end

end