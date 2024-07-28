using DynamicalSystemsBase, Test
using StochasticDiffEq: SDEProblem, SRA, SRI, SOSRA

include("test_system_function.jl")

# Creation of lorenz
@inbounds function lorenz_rule(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end
@inbounds function lorenz_rule_iip(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
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
p0 = [10, 28, 8/3]

# diagonal additive noise
lorenz_oop = CoupledSDEs(lorenz_rule, diagonal_noise(σ), u0, p0)
lorenz_iip = CoupledSDEs(SDEProblem(lorenz_rule_iip, diagonal_noise!(σ), copy(u0), (0.0, Inf), p0))
lorenz_SRA = CoupledSDEs(lorenz_rule, diagonal_noise(σ), u0, p0;
    diffeq = (alg = SRA(), abstol = 1e-6, reltol = 1e-6)
)

for (ds, iip) in zip((lorenz_oop, lorenz_iip, lorenz_SRA), (false, true, false))

    name = (ds === lorenz_SRA) ? "lorSRA" : "lorenz"
    @testset "$(name) IIP=$(iip)" begin
        @test dynamic_rule(ds) == (iip ? lorenz_rule_iip : lorenz_rule)
        test_dynamical_system(ds, u0, p0; idt = false, iip)
    end
end

@testset "correct SDE propagation" begin
    lorenz_oop = CoupledSDEs(lorenz_rule, diagonal_noise(σ), u0, p0)
    @test lorenz_oop.integ.alg isa SOSRA

    lorenz_SRA = CoupledSDEs(lorenz_rule, diagonal_noise(σ), u0, p0;
        diffeq = (alg = SRA(), abstol = 1e-6, reltol = 1e-6, verbose=false)
    )
    @test lorenz_SRA.integ.alg isa SRA
    @test lorenz_SRA.integ.opts.verbose == false

    # # also test SDEproblem creation
    prob = lorenz_SRA.integ.sol.prob

    ds = CoupledSDEs(prob, (alg=SRI(), abstol=0.0, reltol=1e-6, verbose=false))

    @test ds.integ.alg isa SRI
    @test ds.integ.opts.verbose == false

    @test_throws ArgumentError CoupledSDEs(prob; diffeq = (alg=SRI(), ))

end