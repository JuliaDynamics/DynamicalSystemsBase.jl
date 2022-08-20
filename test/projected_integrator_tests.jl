using DynamicalSystemsBase, Test
using LinearAlgebra: norm
@testset "Projected integrator" begin

@testset "Lorenz" begin
    # Set β to 10, we have a stable fixed point
    ds = Systems.lorenz([0.0, 10.0, 1.0]; σ = 10.0, ρ = 28.0, β = 10)
    psys = projected_integrator(ds, 1:2, [0.0])
    @test get_state(psys) == get_state(ds)[1:2]

    reinit!(psys, [1.0, 1.0])
    for j in 1:5
      step!(psys, 1.0)
    end
    y = get_state(psys)
    @test length(y) == 2
    @test abs(sum(y - [16.43, 16.43])) < 0.01

    # Test complete state function
    complete = (y) -> [y[1], y[2], y[2] + 1]
    psys = projected_integrator(ds, 1:2, complete)
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
    complete = y -> 10y
    psys = projected_integrator(ds, projection, complete)
    @test norm(get_state(psys)) == 1
    reinit!(psys, ones(3))
    @test psys.integ.u[1] == 10

    for j in 1:5
      step!(psys, 1.0)
    end
    u = get_state(psys)
    @test abs(sum(u[1:2] - [0.461, 0.461])) < 0.01

    # Trajectory, continuous
    psys = projected_integrator(ds, projection, complete)
    tr = trajectory(psys, 10)
    @test dimension(tr) == 3
    @test norm(tr[end]) == 1

    # Trajectory, discrete
    ds = Systems.towel()
    psys = projected_integrator(ds, [1,2], [0.0])
    @test get_state(psys) == [0.085, -0.121]
    @test dimension(psys) == 2
    tr = trajectory(psys, 10)
    @test dimension(tr) == 2

    p0 = get_parameter(ds, 1)
    @test get_parameter(psys, 1) == p0
    set_parameter!(psys, 1, 0.5)
    @test get_parameter(psys, 1) == 0.5
    set_parameter!(psys, 1, p0)
end


end