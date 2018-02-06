println("\nTesting discrete system types...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays


DS = Systems.henon()
DS2 = Systems.henon_iip()

DS_nojac = DDS(SVector(0.0, 0.0), DS.prob.f, DS.prob.p)
DS2_nojac = DDS([0.0, 0.0], DS2.prob.f, DS2.prob.p)


@testset "isinplace" begin

    @test !isinplace(DS)
    @test !isinplace(DS_nojac)
    @test isinplace(DS2)
    @test isinplace(DS2_nojac)

    @test typeof(DS.J) <: SMatrix
    @test typeof(DS_nojac.J) <: SMatrix
    @test typeof(DS2.J) <: Matrix
    @test typeof(DS2_nojac.J) <: Matrix

end

@testset "Jacobian" begin
    @test jacobian(DS) == jacobian(DS2)
    @test jacobian(DS_nojac) == jacobian(DS2_nojac)

    @test jacobian(DS) ≈ jacobian(DS_nojac)
end

@testset "evolve" begin
    @test evolve(DS) == evolve(DS2) == evolve(DS2_nojac) == evolve(DS_nojac)
    evolve!(DS); evolve!(DS2); evolve!(DS2_nojac); evolve!(DS_nojac)
    @test state(DS) == state(DS2) == state(DS2_nojac) == state(DS_nojac)

    evolve!(DS, 2); evolve!(DS2, 2); evolve!(DS2_nojac, 2); evolve!(DS_nojac, 2)
    @test state(DS) == state(DS2) == state(DS2_nojac) == state(DS_nojac)

    xnew = state(DS)

    set_state!(DS, xnew)
    set_state!(DS2, xnew)

    @test state(DS) == xnew
    @test state(DS2) == xnew

end

@testset "ParallelEvolver" begin
    states = [zeros(2), zeros(2)]

    pe = ParallelEvolver(DS, deepcopy(states))
    pe2 = ParallelEvolver(DS2, states)

    evolve!(pe, 10); evolve!(pe2, 10)
    @test pe.states[1] == pe2.states[1]
    @test pe.states[2] == pe2.states[2]
    @test pe.states[1] == pe.states[2]

    @test !isinf(evolve(DS, 1000000)[1])
end
