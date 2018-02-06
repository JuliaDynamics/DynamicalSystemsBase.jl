println("\nTesting discrete system types...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays


ds = Systems.henon()
ds2 = Systems.henon_iip()

ds_nojac = DDS(SVector(0.0, 0.0), ds.prob.f, ds.prob.p)
ds2_nojac = DDS([0.0, 0.0], ds2.prob.f, ds2.prob.p)


@testset "isinplace" begin

    @test !isinplace(ds)
    @test !isinplace(ds_nojac)
    @test isinplace(ds2)
    @test isinplace(ds2_nojac)

    @test typeof(ds.J) <: SMatrix
    @test typeof(ds_nojac.J) <: SMatrix
    @test typeof(ds2.J) <: Matrix
    @test typeof(ds2_nojac.J) <: Matrix

end

@testset "Jacobian" begin
    @test jacobian(ds) == jacobian(ds2)
    @test jacobian(ds_nojac) == jacobian(ds2_nojac)

    @test jacobian(ds) ≈ jacobian(ds_nojac)
end

@testset "evolve" begin
    @test evolve(ds) == evolve(ds2) == evolve(ds2_nojac) == evolve(ds_nojac)
    evolve!(ds); evolve!(ds2); evolve!(ds2_nojac); evolve!(ds_nojac)
    @test state(ds) == state(ds2) == state(ds2_nojac) == state(ds_nojac)

    xnew = state(ds)

    set_state!(ds, xnew)
    set_state!(ds2, xnew)

    @test state(ds) == xnew
    @test state(ds2) == xnew

end

@testset "ParallelEvolver" begin
    states = [zeros(2), zeros(2)]

    pe = ParallelEvolver(ds, deepcopy(states))
    pe2 = ParallelEvolver(ds2, states)

    evolve!(pe, 10); evolve!(pe2, 10)
    @test pe.states[1] == pe2.states[1]
    @test pe.states[2] == pe2.states[2]
    @test pe.states[1] == pe.states[2]

    @test !isinf(evolve(ds, 1000000)[1])
end
