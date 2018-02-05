
@testset "TangentEvolver" begin

    traj = trajectory(ds, 10)
    jacvals = [-2*p[1]*t[1] for t in traj]

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
