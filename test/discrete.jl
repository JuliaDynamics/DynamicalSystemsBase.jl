using DynamicalSystemsBase, Test

include("test_system_function.jl")

# Creation of Henon map
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
function henon_rule_iip(dx, x, p, n)
    dx .= henon_rule(x, p, n)
    return
end

u0 = zeros(2)
p0 = [1.4, 0.3]

henon_oop = DeterministicIteratedMap(henon_rule, u0, p0)
henon_iip = DeterministicIteratedMap(henon_rule_iip, copy(u0), p0)

@testset "Henon IIP=$(iip)" for (henon, iip) in zip((henon_oop, henon_iip), (false, true))
    @test dynamic_rule(henon) == (iip ? henon_rule_iip : henon_rule)
    test_dynamical_system(henon, u0, p0; idt = true, iip)
end