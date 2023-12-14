using DynamicalSystemsBase, Test

using OrdinaryDiffEq: Vern9, ODEProblem, Rodas5, Tsit5

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

u0 = [0, 10.0, 0]
p0 = [10, 28, 8/3]

lorenz_oop = CoupledODEs(lorenz_rule, u0, p0)
lorenz_iip = CoupledODEs(ODEProblem(lorenz_rule_iip, copy(u0), (0.0, Inf), p0))
lorenz_vern = CoupledODEs(lorenz_rule, u0, p0;
    diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)
)

for (ds, iip) in zip((lorenz_oop, lorenz_iip, lorenz_vern), (false, true, false))

    name = (ds === lorenz_vern) ? "lorvern" : "lorenz"
    @testset "$(name) IIP=$(iip)" begin
        @test dynamic_rule(ds) == (iip ? lorenz_rule_iip : lorenz_rule)
        test_dynamical_system(ds, u0, p0; idt = false, iip)
    end
end

@testset "correct ODE propagation" begin
    lorenz_oop = CoupledODEs(lorenz_rule, u0, p0)
    @test lorenz_oop.integ.alg isa Tsit5

    lorenz_vern = CoupledODEs(lorenz_rule, u0, p0;
        diffeq = (alg = Vern9(), verbose = false, abstol = 1e-9, reltol = 1e-9)
    )
    @test lorenz_vern.integ.alg isa Vern9
    @test lorenz_vern.integ.opts.verbose == false

    # also test ODEproblem creation
    prob = lorenz_vern.integ.sol.prob

    ds = CoupledODEs(prob, (alg=Rodas5(autodiff=false), abstol=0.0, reltol=1e-6, verbose=false))

    @test ds.integ.alg isa Rodas5
    @test ds.integ.opts.verbose == false

    @test_throws ArgumentError CoupledODEs(prob; diffeq = (alg=Rodas5(autodiff=false), ))

end