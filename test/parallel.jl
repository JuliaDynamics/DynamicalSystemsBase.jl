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

u0 = ones(2)
states = [u0, u0 .+ 0.01, deepcopy(u0)]

p0_disc = [1.1, 0.8]

trivial_disc_oop = DeterministicIteratedMap(trivial_rule, u0, p0_disc)
trivial_disc_iip = DeterministicIteratedMap(trivial_rule_iip, copy(u0), p0_disc)

pds_disc_oop = ParallelDynamicalSystem(trivial_disc_oop, states)
pds_disc_iip = ParallelDynamicalSystem(trivial_disc_iip, deepcopy(states))

p0_cont = [0.1, -0.4]

trivial_cont_oop = CoupledODEs(trivial_rule, u0, p0_cont)
trivial_cont_iip = CoupledODEs(trivial_rule_iip, copy(u0), p0_cont)

pds_cont_oop = ParallelDynamicalSystem(trivial_cont_oop, states)
pds_cont_iip = ParallelDynamicalSystem(trivial_cont_iip, deepcopy(states))

lmax_disc = log(1.1)
lmax_cont = 0.1

function parallel_integration_tests(pdsa)
    @testset "analytic parallel" begin
    # uses knowledge of trivial rule
    reinit!(pdsa)
    @test current_state(pdsa, 1) == current_state(pdsa, 3) == initial_state(pdsa, 1)
    @test current_state(pdsa, 1) != current_state(pdsa, 2)
    dmax1 = abs(current_state(pdsa, 1)[1] - current_state(pdsa, 2)[1])
    emax1 = abs(current_state(pdsa, 1)[2] - current_state(pdsa, 2)[2])
    step!(pdsa)
    @test current_state(pdsa, 1) == current_state(pdsa, 3) != current_state(pdsa, 2)
    step!(pdsa, 2)
    @test current_state(pdsa, 1) == current_state(pdsa, 3) != current_state(pdsa, 2)
    dmax2 = abs(current_state(pdsa, 1)[1] - current_state(pdsa, 2)[1])
    emax2 = abs(current_state(pdsa, 1)[2] - current_state(pdsa, 2)[2])
    @test dmax2 > dmax1 # unstable first dimension
    @test emax2 < emax1 # stable second dimension

    # Check that `set_state!` works as expected
    set_state!(pdsa, deepcopy(current_state(pdsa, 1)), 2)
    @test current_state(pdsa) == current_state(pdsa, 2)
    step!(pdsa, 2)
    @test current_state(pdsa) == current_state(pdsa, 2)
    end
end

function max_lyapunov_test(pdsa, lmax)
    @testset "max lyapunov" begin
    reinit!(pdsa)
    # Quick and dirt code of estimating max Lyapunov
    current_norm(pdsa) = norm(current_state(pdsa, 1) .- current_state(pdsa, 2))
    d0 = current_norm(pdsa)
    t0 = initial_time(pdsa)
    λ = zero(d0)
    T = 300
    for _ in 1:T
        step!(pdsa, 1)
        d = current_norm(pdsa)
        λ += log(d/d0)
        # here one would do a rescale, but we won't; we don't evolve enough to matter
        # local lyapunov exponent is the relative distance of the trajectories
        d0 = d
    end
    # Do final calculation, only useful for continuous system
    d = current_norm(pdsa)
    λ += log(d/d0)
    final_λ = λ/(current_time(pdsa) - t0)
    # @show final_λ
    @test final_λ ≈ lmax atol = 1e-2 rtol = 0
    end
end

for (ds, idt, iip) in zip(
        (pds_disc_oop, pds_disc_iip, pds_cont_oop, pds_cont_iip,),
        (true, true, false, false), (false, true, false, true),
    )

    @testset "idt=$(idt), iip=$(iip)" begin
        p0 = idt ? p0_disc : p0_cont
        test_dynamical_system(ds, u0, p0; idt, iip = true, test_trajectory = false)
        parallel_integration_tests(ds)
        max_lyapunov_test(ds, idt ? lmax_disc : lmax_cont)
    end
end

# Generic Parallel
@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
end
function duffing_rule_iip(du, u, p, t)
    du .= duffing_rule(u, p, t)
    return nothing
end

u0 = [0.1, 0.25]
p0 = [1.0, 0.3, 0.2, -1]
T = 2π/1.0

duffing_oop = StroboscopicMap(CoupledODEs(duffing_rule, u0, p0), T)
duffing_iip = StroboscopicMap(T, duffing_rule_iip, copy(u0), p0)

states = [u0, u0 .+ 0.01]

pds_cont_oop = ParallelDynamicalSystem(duffing_oop, states)
pds_cont_iip = ParallelDynamicalSystem(duffing_iip, deepcopy(states))

@testset "parallel duffing IIP=$iip" for (ds, iip) in zip((pds_cont_oop, pds_cont_iip,), (true, false))
    test_dynamical_system(ds, u0, p0; idt = true, iip = true, test_trajectory = false)
end

# TODO: Test that Lyapunovs of this match the original system
# But test this in ChaosTools.jl