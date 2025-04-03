using DynamicalSystemsBase, Test

function oop(u, p, t)
    return p[1] * SVector(u[1], -u[2])
end

function iip(du, u, p, t)
    du .= oop(u, p, t)
    return nothing
end


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

    eqs = [D(u[1]) ~ 3.0 * u[1],
           D(u[2]) ~ -3.0 * u[2]]
    @named sys = ODESystem(eqs, t)
    sys = structural_simplify(sys)

    prob = ODEProblem(sys, [1.0, 1.0], (0.0, 1.0); jac=true)
    ds = CoupledODEs(prob)

    jac = jacobian(ds)
    @test jac([1.0, 1.0], [], 0.0) == [3 0;0 -3]

    @testset "CoupledSDEs" begin
        # just to check if MTK @brownian does not give any problems
        using StochasticDiffEq
        @brownian β
        eqs = [D(u[1]) ~ 3.0 * u[1]+ β,
                D(u[2]) ~ -3.0 * u[2] + β]
        @mtkbuild sys = System(eqs, t)

        prob = SDEProblem(sys, [1.0, 1.0], (0.0, 1.0), jac=true)
        sde = CoupledSDEs(prob)

        jac = jacobian(sde)
        @test jac.jac_oop isa RuntimeGeneratedFunction
        @test jac([1.0, 1.0], [], 0.0) == [3 0;0 -3]
    end
end
