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