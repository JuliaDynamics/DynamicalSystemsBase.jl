using DynamicalSystemsBase, Test
using OrdinaryDiffEqTsit5: Tsit5
using StochasticDiffEq: SDEProblem, SRA, SOSRA, LambaEM, CorrelatedWienerProcess, EM

StochasticSystemsBase = Base.get_extension(DynamicalSystemsBase, :StochasticSystemsBase)
diffusion_matrix = StochasticSystemsBase.diffusion_matrix

# Creation of lorenz
@inbounds function lorenz_rule(u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du1 = σ * (u[2] - u[1])
    du2 = u[1] * (ρ - u[3]) - u[2]
    du3 = u[1] * u[2] - β * u[3]
    return SVector{3}(du1, du2, du3)
end
@inbounds function lorenz_rule_iip(du, u, p, t)
    σ = p[1]
    ρ = p[2]
    β = p[3]
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

σ = 0.1
function diagonal_noise!(σ)
    return function (du, u, p, t)
        du .= σ .* ones(length(u))
        return nothing
    end
end
diagonal_noise(σ) = (u, p, t) -> SVector{3}(σ, σ, σ)

u0 = [0, 10.0, 0]
p0 = [10, 28, 8 / 3]
Γ = [1.0 0.3 0.0; 0.3 1 0.5; 0.0 0.5 1.0]

# diagonal additive noise
lor_oop = CoupledSDEs(lorenz_rule, u0, p0)
lor_iip = CoupledSDEs(
    SDEProblem(
        lorenz_rule_iip, diagonal_noise!(σ), copy(u0), (0.0, Inf), p0
    )
)
lor_SRA = CoupledSDEs(
    lorenz_rule, u0, p0;
    diffeq = (alg = SRA(), abstol = 1.0e-2, reltol = 1.0e-2)
)

diffeq_cov = (alg = LambaEM(), abstol = 1.0e-2, reltol = 1.0e-2, dt = 0.1)
lor_oop_cov = CoupledSDEs(lorenz_rule, u0, p0; covariance = Γ, diffeq = diffeq_cov)
lor_iip_cov = CoupledSDEs(lorenz_rule_iip, u0, p0; covariance = Γ, diffeq = diffeq_cov)

for (ds, iip) in zip(
        (lor_oop, lor_iip, lor_SRA, lor_oop_cov, lor_iip_cov), (false, true, false, false, true)
    )
    name = (ds === lor_SRA) ? "lorvern" : "lorenz"
    @testset "$(name) IIP=$(iip)" begin
        @test dynamic_rule(ds) == (iip ? lorenz_rule_iip : lorenz_rule)
        test_dynamical_system(ds, u0, p0; idt = false, iip = iip)
    end
end

@testset "correct SDE propagation" begin
    lorenz_oop = CoupledSDEs(lorenz_rule, u0, p0)
    @test lorenz_oop.integ.alg isa SOSRA

    lorenz_SRA = CoupledSDEs(
        lorenz_rule, u0, p0;
        diffeq = (alg = SRA(), abstol = 1.0e-3, reltol = 1.0e-3, verbose = false)
    )
    @test lorenz_SRA.integ.alg isa SRA
    # SciML moved from Bool verbose to a `DEVerbosity` struct of per-toggle verbosities.
    @test nameof(typeof(lorenz_SRA.integ.opts.verbose.linear_verbosity)) == :None

    # also test SDEproblem creation
    prob = lorenz_SRA.integ.sol.prob

    ds = CoupledSDEs(prob, (alg = SRA(), abstol = 0.0, reltol = 1.0e-3, verbose = false))

    @test ds.integ.alg isa SRA
    @test nameof(typeof(ds.integ.opts.verbose.linear_verbosity)) == :None

    @test_throws ArgumentError CoupledSDEs(prob; diffeq = (alg = SRA(),))

    # CoupledODEs creation
    ds = CoupledODEs(lorenz_oop)
    @test dynamic_rule(ds).f == lorenz_rule
    @test ds.integ.alg isa Tsit5
    test_dynamical_system(ds, u0, p0; idt = false, iip = false)
    # and back
    sde = CoupledSDEs(ds, p0)
    @test dynamic_rule(sde).f.f == lorenz_rule
    @test sde.integ.alg isa SOSRA
end

@testset "interface" begin
    f(u, p, t) = 1.01u
    f!(du, u, p, t) = du .= 1.01u
    @testset "covariance" begin
        g(u, p, t) = sqrt([1 0.3; 0.3 1])
        corr = CoupledSDEs(f, zeros(2); covariance = [1 0.3; 0.3 1])
        corr_alt = CoupledSDEs(f, zeros(2); g = g, noise_prototype = zeros(2, 2))
        @test corr.noise_type == corr_alt.noise_type
        @test all(
            DynamicalSystemsBase.referrenced_sciml_prob(corr).g(zeros(2), (), 0.0) .==
                DynamicalSystemsBase.referrenced_sciml_prob(corr_alt).g(zeros(2), (), 0.0)
        )
    end

    @testset "ArgumentError" begin
        W = CorrelatedWienerProcess([1 0.3; 0.3 1], 0.0, zeros(2), zeros(2))
        @test_throws ArgumentError CoupledSDEs(f!, zeros(2); noise_process = W)

        g!(du, u, p, t) = du .= u
        @test_throws ArgumentError CoupledSDEs(
            f!, zeros(2); g = g!, covariance = [1 0.3; 0.3 1]
        )

        g(u, p, t) = u
        @test_throws AssertionError CoupledSDEs(f!, zeros(2); g = g)

        Csde = CoupledSDEs(f!, zeros(2))
        diffeq = (alg = SRA(), abstol = 1.0e-2, reltol = 1.0e-2)
        @test_throws ArgumentError CoupledSDEs(Csde.integ.sol.prob; diffeq = diffeq)
    end
end

@testset "utilities" begin
    StochasticSystemsBase = Base.get_extension(DynamicalSystemsBase, :StochasticSystemsBase)
    diffusion_matrix = StochasticSystemsBase.diffusion_matrix

    @testset "diffusion_matrix" begin
        Γ = [1.0 0.3 0.0; 0.3 1 0.5; 0.0 0.5 1.0]
        A = sqrt(Γ)
        lorenz_oop = CoupledSDEs(lorenz_rule, u0, p0, covariance = Γ, diffeq = diffeq_cov)
        @test A ≈ diffusion_matrix(lorenz_oop)
        @test A isa AbstractMatrix
        @test Γ ≈ A * A'
    end

    @testset "parametrize cov" begin
        Γ = [1.0 0.3; 0.3 1]
        f(u, p, t) = [0.0, 0.0]
        function diffusion(u, p, t)
            Γ = [1.0 p[1]; p[1] 1.0]
            sqrt(Γ)
        end
        ds = CoupledSDEs(f, zeros(2), [0.3]; g = diffusion, noise_prototype = zeros(2, 2))
        A = diffusion_matrix(ds)
        @test Γ ≈ A * A'
        set_parameter!(ds, 1, 0.5)
        A = diffusion_matrix(ds)
        @test A * A' ≈ [1.0 0.5; 0.5 1.0]
    end
    @testset "approximate cov" begin
        Γ = [1.0 0.3; 0.3 1]
        f(u, p, t) = [0.0, 0.0]
        diffeq_cov = (alg = EM(), abstol = 1.0e-2, reltol = 1.0e-2, dt = 0.1)

        ds = CoupledSDEs(f, zeros(2), (); covariance = Γ, diffeq = diffeq_cov)
        tr, _ = trajectory(ds, 10_000, Δt = 0.1)
        approx = cov(diff(reduce(hcat, tr.data), dims = 2) ./ sqrt(0.1), dims = 2)
        @test approx ≈ Γ atol = 1.0e-1
    end
end

@testset "seed" begin
    f(u, p, t) = -u

    @testset "constructor default is randomized" begin
        # Two CoupledSDEs without explicit seed get different seeds and different trajectories.
        ds1 = CoupledSDEs(f, [1.0])
        ds2 = CoupledSDEs(f, [1.0])
        @test ds1.integ.sol.prob.seed != 0
        @test ds1.integ.sol.prob.seed != ds2.integ.sol.prob.seed
        step!(ds1, 1.0); step!(ds2, 1.0)
        @test current_state(ds1) != current_state(ds2)
    end

    @testset "constructor explicit seed is reproducible" begin
        seed = UInt64(20250502)
        ds1 = CoupledSDEs(f, [1.0]; seed = seed)
        ds2 = CoupledSDEs(f, [1.0]; seed = seed)
        step!(ds1, 1.0); step!(ds2, 1.0)
        @test current_state(ds1) ≈ current_state(ds2)
    end

    @testset "reinit! default reseeds with fresh randomness" begin
        ds = CoupledSDEs(f, [1.0]; seed = UInt64(1))
        reinit!(ds); step!(ds, 1.0); ua = copy(current_state(ds))
        reinit!(ds); step!(ds, 1.0); ub = copy(current_state(ds))
        # Different default seeds → different trajectories
        @test ua != ub
    end

    @testset "reinit! accepts explicit seed kwarg" begin
        ds = CoupledSDEs(f, [1.0])
        # Same explicit seed → same noise stream after reinit
        reinit!(ds; seed = UInt64(42)); step!(ds, 1.0); ua = copy(current_state(ds))
        reinit!(ds; seed = UInt64(42)); step!(ds, 1.0); ub = copy(current_state(ds))
        @test ua ≈ ub
        # Different explicit seed → different trajectory
        reinit!(ds; seed = UInt64(43)); step!(ds, 1.0); uc = copy(current_state(ds))
        @test ua != uc
# Regression test for https://github.com/JuliaDynamics/DynamicalSystemsBase.jl/issues/251:
# the auto-generated diffusion closure used to recompute its (constant) output on every
# call, allocating ~1 KB per `step!` and dominating long integrations. The closure must
# now return a precomputed constant with no allocations.
@testset "auto-diffusion closure is allocation-free (#251)" begin
    f_oop(u, p, t) = SVector{2}(0.0, 0.0)
    f_iip(du, u, p, t) = (du .= 0; nothing)
    Γ = [1.0 0.3; 0.3 1.0]

    function call_oop(g, n)
        s = SVector(0.1, 0.2)
        out = SVector(0.0, 0.0)
        for _ in 1:n
            out = g(s, nothing, 0.0)
        end
        return out
    end
    function call_iip!(g, du, n)
        u = [0.0, 0.0]
        for _ in 1:n
            g(du, u, nothing, 0.0)
        end
        return du
    end

    @testset "OOP diagonal" begin
        ds = CoupledSDEs(f_oop, SVector(0.0, 0.0); noise_strength = 2.5)
        g = ds.integ.sol.prob.f.g
        @test g(SVector(0.0, 0.0), nothing, 0.0) === g(SVector(1.0, 1.0), nothing, 5.0)
        @test g(SVector(0.0, 0.0), nothing, 0.0) == SVector(2.5, 2.5)
        call_oop(g, 10) # warmup
        @test (@allocated call_oop(g, 10_000)) < 100
    end

    @testset "OOP non-diagonal" begin
        ds = CoupledSDEs(f_oop, SVector(0.0, 0.0); covariance = Γ, noise_strength = 1.5)
        g = ds.integ.sol.prob.f.g
        @test g(SVector(0.0, 0.0), nothing, 0.0) === g(SVector(1.0, 1.0), nothing, 5.0)
        @test g(SVector(0.0, 0.0), nothing, 0.0) ≈ 1.5 .* sqrt(Γ)
        call_oop(g, 10)
        @test (@allocated call_oop(g, 10_000)) < 100
    end

    @testset "IIP diagonal" begin
        ds = CoupledSDEs(f_iip, [0.0, 0.0]; noise_strength = 2.5)
        g = ds.integ.sol.prob.f.g
        du = zeros(2)
        g(du, [0.0, 0.0], nothing, 0.0)
        @test du == [2.5, 2.5]
        call_iip!(g, du, 10)
        @test (@allocated call_iip!(g, du, 10_000)) < 100
    end

    @testset "IIP non-diagonal" begin
        ds = CoupledSDEs(f_iip, [0.0, 0.0]; covariance = Γ, noise_strength = 1.5)
        g = ds.integ.sol.prob.f.g
        DU = zeros(2, 2)
        g(DU, [0.0, 0.0], nothing, 0.0)
        @test DU ≈ 1.5 .* sqrt(Γ)
        function call_iip_mat!(g, DU, n)
            u = [0.0, 0.0]
            for _ in 1:n
                g(DU, u, nothing, 0.0)
            end
            return DU
        end
        call_iip_mat!(g, DU, 10)
        @test (@allocated call_iip_mat!(g, DU, 10_000)) < 100
    end
end
