using DynamicalSystemsBase, Test
using OrdinaryDiffEq: Tsit5
using StochasticDiffEq: SDEProblem, SRA, SOSRA, LambaEM, CorrelatedWienerProcess

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
    function (du, u, p, t)
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
lor_iip = CoupledSDEs(SDEProblem(
    lorenz_rule_iip, diagonal_noise!(σ), copy(u0), (0.0, Inf), p0))
lor_SRA = CoupledSDEs(lorenz_rule, u0, p0;
    diffeq = (alg = SRA(), abstol = 1e-2, reltol = 1e-2)
)

diffeq_cov = (alg = LambaEM(), abstol = 1e-2, reltol = 1e-2, dt=0.1)
lor_oop_cov = CoupledSDEs(lorenz_rule, u0, p0; covariance = Γ, diffeq=diffeq_cov)
lor_iip_cov = CoupledSDEs(lorenz_rule_iip, u0, p0; covariance = Γ, diffeq=diffeq_cov)

for (ds, iip) in zip(
    (lor_oop, lor_iip, lor_SRA, lor_oop_cov, lor_iip_cov), (false, true, false, false, true))
    name = (ds === lor_SRA) ? "lorvern" : "lorenz"
    @testset "$(name) IIP=$(iip)" begin
        @test dynamic_rule(ds) == (iip ? lorenz_rule_iip : lorenz_rule)
        test_dynamical_system(ds, u0, p0; idt = false, iip = iip)
    end
end

@testset "correct SDE propagation" begin
    lorenz_oop = CoupledSDEs(lorenz_rule, u0, p0)
    @test lorenz_oop.integ.alg isa SOSRA

    lorenz_SRA = CoupledSDEs(lorenz_rule, u0, p0;
        diffeq = (alg = SRA(), abstol = 1e-3, reltol = 1e-3, verbose = false)
    )
    @test lorenz_SRA.integ.alg isa SRA
    @test lorenz_SRA.integ.opts.verbose == false

    # also test SDEproblem creation
    prob = lorenz_SRA.integ.sol.prob

    ds = CoupledSDEs(prob, (alg = SRA(), abstol = 0.0, reltol = 1e-3, verbose = false))

    @test ds.integ.alg isa SRA
    @test ds.integ.opts.verbose == false

    @test_throws ArgumentError CoupledSDEs(prob; diffeq = (alg = SRA(),))

    # CoupledODEs creation
    ds = CoupledODEs(lorenz_oop)
    @test dynamic_rule(ds) == lorenz_rule
    @test ds.integ.alg isa Tsit5
    test_dynamical_system(ds, u0, p0; idt = false, iip = false)
    # and back
    sde = CoupledSDEs(ds, p0)
    @test dynamic_rule(sde) == lorenz_rule
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
        @test all(corr.integ.g(zeros(2), (), 0.0) .== corr_alt.integ.g(zeros(2), (), 0.0))
    end

    @testset "ArgumentError" begin
        W = CorrelatedWienerProcess([1 0.3; 0.3 1], 0.0, zeros(2), zeros(2))
        @test_throws ArgumentError CoupledSDEs(f!, zeros(2); noise_process = W)

        g!(du, u, p, t) = du .= u
        @test_throws ArgumentError CoupledSDEs(
            f!, zeros(2); g = g!, covariance = [1 0.3; 0.3 1])

        g(u, p, t) = u
        @test_throws AssertionError CoupledSDEs(f!, zeros(2); g = g)

        Csde = CoupledSDEs(f!, zeros(2))
        diffeq = (alg = SRA(), abstol = 1e-2, reltol = 1e-2)
        @test_throws ArgumentError CoupledSDEs(Csde.integ.sol.prob; diffeq = diffeq)
    end
end

@testset "utilities" begin
    StochasticSystemsBase = Base.get_extension(DynamicalSystemsBase, :StochasticSystemsBase)
    diffusion_matrix = StochasticSystemsBase.diffusion_matrix

    @testset "diffusion_matrix" begin
        Γ = [1.0 0.3 0.0; 0.3 1 0.5; 0.0 0.5 1.0]
        A = sqrt(Γ)
        lorenz_oop = CoupledSDEs(lorenz_rule, u0, p0, covariance = Γ, diffeq=diffeq_cov)
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
        ds = CoupledSDEs(f, zeros(2), (); covariance = Γ, diffeq=diffeq_cov)
        tr, _ = trajectory(ds, 1_000)
        approx = cov(diff(reduce(hcat, tr.data), dims=2), dims=2)
        @test approx ≈ Γ atol=1e-1 broken = true
        # I think I understand something wromg here
    end
end
