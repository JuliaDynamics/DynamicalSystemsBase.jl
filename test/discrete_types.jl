println("\nTesting discrete system types...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

# out of place:
p = [1.4, 0.3]
@inline henon_eom(x, p) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
@inline henon_jacob(x, p) = @SMatrix [-2*p[1]*x[1] 1.0; p[2] 0.0]

# inplace:
function henon_eom_iip(dx, x, p)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end
function henon_jacob_iip(J, x, p)
    J[1,1] = -2*p[1]*x[1]
    J[1,2] = 1.0
    J[2,1] = p[2]
    J[2,2] = 0.0
    return
end

ds_nojac = DDS(SVector(0.0, 0.0), henon_eom, p)
ds = DDS(SVector(0.0, 0.0), henon_eom, p, henon_jacob)

ds2 = DDS([0.0, 0.0], henon_eom_iip, p, henon_jacob_iip)
ds2_nojac = DDS([0.0, 0.0], henon_eom_iip, p)

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

    pe = ParallelEvolver(ds, states)
    pe2 = ParallelEvolver(ds2, states)


    evolve!(pe, 10); evolve!(pe2, 10)
    @test pe.states[1] == pe2.states[1]
    @test pe.states[2] == pe2.states[2]
    @test pe.states[1] == pe.states[2]

    @test !isinf(evolve(ds, 1000000)[1])
end

@testset "TangentEvolver" begin

    ws = orthonormal(2, 2)

    te = TangentEvolver(ds, ws)
    te2 = TangentEvolver(ds2, ws)
    te_nojac = TangentEvolver(ds_nojac, ws)
    te2_nojac = TangentEvolver(ds2_nojac, ws)

    evolve!(te, 2)
    evolve!(te2, 2)
    evolve!(te_nojac, 2)
    evolve!(te2_nojac, 2)

    @test te.state == te2.state
    @test te.ws == te2.ws
    @test te2_nojac.ws == te_nojac.ws
    @test te.ws ≈ te_nojac.ws

    s = evolve(ds, 2)

    @test state(te) == s
    @test state(te2_nojac) == s
end
