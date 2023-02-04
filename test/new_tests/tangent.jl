using DynamicalSystemsBase, Test
using LinearAlgebra: norm

include("test_system_function.jl")

# Creation of a trivial system with one coordinate going to infinity
# and the other to zero. Lyapunov exponents are known analytically
trivial_rule(x, p, n) = SVector{2}(p[1]*x[1], p[2]*x[2])
function trivial_rule_iip(dx, x, p, n)
    dx .= trivial_rule(x, p, n)
    return
end
trivial_jac(x, p, n) = SMatrix{2, 2}(p[1], 0, 0, p[2])
trivial_jac_iip(J, x, p, n) = (J[1,1] = p[1]; J[2,2] = p[2]; nothing)

u0 = ones(2)
p0_disc = [1.1, 0.8]
p0_cont = [0.1, -0.4]

# Test suite explicitly for tangent space
# if IDT: after one step deviation vectors become the Jacobian
# and then square.
# If not IDT: step! for exactly 1, then deviations become exp(λ)
# actually, step for 1 and always become exp(λ)

# Allright, unfortunately here we have to test a ridiculous amount of multiplicity...
@testset "IDT=$(IDT), IIP=$(IIP), IAD=$(IAD)" for IDT in (true, false), IIP in (false, true), IAD in (false, true)
    SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
    rule = IIP ? trivial_rule_iip : trivial_rule
    p0 = IDT ? p0_disc : p0_cont
    lyapunovs = IDT ? log.(p0) : p0
    Jf = IAD ? nothing : (IIP ? trivial_jac_iip : trivial_jac)

    ds = SystemType(rule, u0, p0)
    tands = TangentDynamicalSystem(ds; J = Jf)

    test_dynamical_system(tands, u0, p0, "tangent"; idt=IDT, iip=IIP, test_trajectory = false)

end
