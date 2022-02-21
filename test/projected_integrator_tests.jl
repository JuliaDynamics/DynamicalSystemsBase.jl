using LinearAlgebra: norm

@testset "Projected integrator" begin

@testset "Lorenz" begin
    # Set β to 10, we have a stable fixed point
    ds = Systems.lorenz([0.0, 10.0, 1.0]; σ = 10.0, ρ = 28.0, β = 10)
    psys = projected_integrator(ds, 1:2, [0.0])

    reinit!(psys, [1.0, 1.0])
    for j in 1:5
      step!(psys, 1.0)
    end
    y = get_state(psys)
    @test length(y) == 2
    @test abs(sum(y - [16.43, 16.43])) < 0.01

    # Test complete state function
    pfun(y) = [y[1], y[2], y[2] + 1]
    psys = projectedintegrator(ds, 1:2, complete_state)
    reinit!(psys, [2.0, 1.0])
    y = get_state(psys)
    @test y[1] == 2
    @test length(y) == 2

    for j in 1:5
      step!(psys, 1.0)
    end
    u = get_state(psys)
    @test abs(sum(u - [16.43, 16.43])) < 0.01

    # Test projection function on a sphere of unit radius
    projection(u) = u/norm(u)
    complete_state = y -> 10y
    psys = projectedintegrator(ds, projection, complete_state)
    @test norm(get_state(psys)) == 1
    reinit!(psys, ones(3))
    @test psys.integ.u[1] == 10

    for j in 1:5
      step!(psys, 1.0)
    end
    u = get_state(psys)
    @test abs(sum(u[1:2] - [0.461, 0.461])) < 0.01
end


end