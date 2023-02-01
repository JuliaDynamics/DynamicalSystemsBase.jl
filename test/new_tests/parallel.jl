using DynamicalSystemsBase, Test

include("test_system_function.jl")

# @testset "Discrete (henon)"

# Creation of a trivial system with one coordinate going to infinity
# and the other to zero. Exponents are the logs of coefficients
trivial_discrete_rule(x, p, n) = SVector{2}(1.1x[1], 0.9x[2])
function trivial_discrete_rule_iip(dx, x, p, n)
    dx .= trivial_discrete_rule(x, p, n)
    return
end

u0 = ones(2)

trivial_oop = DeterministicIteratedMap(trivial_discrete_rule, u0)
trivial_iip = DeterministicIteratedMap(trivial_discrete_rule_iip, copy(u0))

states = [u0, u0 .+ 0.01, deepcopy(u0)]

pds_oop = ParallelDynamicalSystem(trivial_oop, states)