println("\nTesting tangent space evolution...")
if current_module() != DynamicalSystemsBase
    using DynamicalSystemsBase
end
using Base.Test, StaticArrays


ds = Systems.henon()
ds2 = Systems.henon_iip()

ds_nojac = DDS(SVector(0.0, 0.0), ds.prob.f, ds.prob.p)
ds2_nojac = DDS([0.0, 0.0], ds2.prob.f, ds2.prob.p)

@testset "TangentEvolver" begin

    traj = trajectory(ds, 10)
    jacvals = [-2*ds.prob.p[1]*t[1] for t in traj]

    ws = orthonormal(2, 2)

    te = TangentEvolver(ds, ws)
    te2 = TangentEvolver(ds2, deepcopy(ws))
    te_nojac = TangentEvolver(ds_nojac, deepcopy(ws))
    te2_nojac = TangentEvolver(ds2_nojac, deepcopy(ws))

    evolve!(te)
    evolve!(te2)
    evolve!(te_nojac)
    evolve!(te2_nojac)

    @test te.state == te2.state
    @test te_nojac.state == te2_nojac.state
    @test te.state == te2_nojac.state

    @test te.ws == te2.ws
    @test te_nojac.ws ≈ te.ws
    @test te2_nojac.ws ≈ te.ws
    @test te2_nojac.ws == te_nojac.ws
    @test te.ws ≈ te2_nojac.ws

    evolve!(te, 5)
    evolve!(te2, 5)
    evolve!(te_nojac, 5)
    evolve!(te2_nojac, 5)

    @test te2.ds.J[1] ≈ jacvals[6]
    @test te2_nojac.ds.J[1] ≈ jacvals[6]

    @test te.state == te2.state
    @test te_nojac.state == te2_nojac.state
    @test te.state == te2_nojac.state

    @test te.ws == te2.ws
    @test te2_nojac.ws == te_nojac.ws
    @test te.ws ≈ te_nojac.ws

    s = evolve(ds, 6)

    @test state(te) == s
    @test state(te2_nojac) == s
end

# Add small lyapunov test here
D = dimension(ds)
T = eltype(ds)
N = 10000
@testset "lyapunovs $v" for v in ["oop", "oop nojac", "iip", "iip nojac"]
    λ = @SVector zeros(T, D)
    Q = @SMatrix eye(T, D)
    te = if v == "oop"
        TangentEvolver(ds, D)
    elseif v == "oop nojac"
        TangentEvolver(ds_nojac, D)
    elseif v == "iip"
        TangentEvolver(ds2, D)
    elseif v == "iip nojac"
        TangentEvolver(ds2_nojac, D)
    end
    set_tangent!(te, Q)
    for i in 1:N
        set_tangent!(te, Q)
        evolve!(te)
        Q, R = qr(te.ws)
        λ += log.(abs.(diag(R)))
    end
    λ = λ/N
    @test 0.418 < λ[1] < 0.422
    @test -1.63 < λ[2] < -1.61
end
