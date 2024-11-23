using DynamicalSystemsBase, Test

function oop(u, p, t)
    return p[1] * SVector(u[1], -u[2])
end

function iip(du, u, p, t)
    du .= oop(u, p, t)
    return nothing
end

#%%
@testset "IDT=$(IDT), IIP=$(IIP)" for IDT in (true, false), IIP in (false, true)
    SystemType = IDT ? DeterministicIteratedMap : CoupledODEs
    rule = IIP ? iip : oop
    p = 3.0
    u0 = [1.0, 1.0]
    result = [p 0.0; 0.0 -p]

    ds = SystemType(rule, u0, p)
    J0 = zeros(dimension(ds), dimension(ds))
    J = jacobian(ds)
    if IIP
        J(J0, current_state(ds), current_parameters(ds), 0.0)
        @test J0 == result
    else
        @test J(current_state(ds), current_parameters(ds), 0.0) == result
    end
end

@testset "MTK Jacobian" begin
    using ModelingToolkit
    using ModelingToolkit: Num, RuntimeGeneratedFunctions.RuntimeGeneratedFunction
    using DynamicalSystemsBase: SciMLBase
    @independent_variables t
    @variables u(t)[1:2]
    D = Differential(t)
    p = 3.0

    eqs = [D(u[1]) ~ p * u[1],
        D(u[2]) ~ -p * u[2]]

    @named sys = ODESystem(eqs, t)
    sys = structural_simplify(sys)

    jac = calculate_jacobian(sys)
    @test jac isa Matrix{Num}

    prob = ODEProblem(sys, [1.0, 1.0], (0.0, 1.0); jac=true)
    ode = CoupledODEs(prob)
    @test ode.integ.f.jac.jac_oop isa RuntimeGeneratedFunction
    @test ode.integ.f.jac([1.0, 1.0], [3.0], 0.0) isa Matrix{Float64}

    jac = jacobian(ode)
    @test jac.jac_oop isa RuntimeGeneratedFunction
    @test jac([1.0, 1.0], [3.0], 0.0) isa Matrix{Float64}

    @testset "CoupledSDEs" begin
        using StochasticDiffEq
        @brownian β
        eqs = [D(u[1]) ~ p * u[1]+ β,
                D(u[2]) ~ -p * u[2] + β]
        @mtkbuild sys = System(eqs, t)

        jac = calculate_jacobian(sys)
        @test jac isa Matrix{Num}

        prob = SDEProblem(sys, [1.0, 1.0], (0.0, 1.0), jac=true)
        sde = CoupledSDEs(prob)
        @test sde.integ.f.jac.jac_oop isa RuntimeGeneratedFunction
        @test sde.integ.f.jac([1.0, 1.0], [3.0], 0.0) isa Matrix{Float64}

        jac = jacobian(ode)
        @test jac.jac_oop isa RuntimeGeneratedFunction
        @test jac([1.0, 1.0], [3.0], 0.0) isa Matrix{Float64}
    end
end
