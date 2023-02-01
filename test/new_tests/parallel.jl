using DynamicalSystemsBase, Test

include("test_system_function.jl")

# @testset "Discrete (henon)"

# Creation of a trivial system with one coordinate going to infinity
# and the other to zero. Exponents are the logs of coefficients
trivial_discrete_rule(x, p, n) = SVector{2}(p[1]*x[1], p[2]*x[2])
function trivial_discrete_rule_iip(dx, x, p, n)
    dx .= trivial_discrete_rule(x, p, n)
    return
end

u0 = ones(2)
p0 = [1.1, 0.9]
trivial_oop = DeterministicIteratedMap(trivial_discrete_rule, u0, p0)
trivial_iip = DeterministicIteratedMap(trivial_discrete_rule_iip, copy(u0), p0)

states = [u0, u0 .+ 0.01, deepcopy(u0)]

pds_oop = ParallelDynamicalSystem(trivial_oop, states)
pds_iip = ParallelDynamicalSystem(trivial_iip, states)

name = "parallel discrete"
test_dynamical_system(pds_oop, u0, p0, name; idt = true, iip = true, test_trajectory = false)

# function parallel_integration_tests(pdsa)
#     reinit!(pdsa)
#     @test current_state(pdsa, 1) == current_state(pdsa, 3) == initial_state(pdsa, 1)
#     @test current_state(pdsa, 1) != current_state(pdsa, 2)
#     step!(pdsa)
#     @test current_state(pdsa, 1) == current_state(pdsa, 3) != current_state(pdsa, 2)
#     step!(pdsa, 2)
#     @test current_state(pdsa, 1) == current_state(pdsa, 3) != current_state(pdsa, 2)
# end



# for (ds, iip) in zip((lorenz_oop, lorenz_iip, lorenz_vern), (false, true, false))

#     @test dynamic_rule(ds) == (iip ? lorenz_rule_iip : lorenz_rule)
#     name = (ds === lorenz_vern) ? "lorvern" : "lorenz"
#     test_dynamical_system(ds, u0, name, false, iip)

# end
