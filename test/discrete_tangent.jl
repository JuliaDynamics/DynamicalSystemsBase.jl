println("\nTesting tangent space evolution...")
if current_module() != DynamicalSystemsBase
    using DynamicalSystemsBase
end
using Base.Test, StaticArrays

DS = Systems.henon()
DS2 = Systems.henon_iip()

DS_nojac = DDS(SVector(0.0, 0.0), DS.prob.f; p = DS.prob.p)
DS2_nojac = DDS([0.0, 0.0], DS2.prob.f; p = DS2.prob.p)


@testset "TangentEvolver" begin

    traj = trajectory(DS, 10)
    jacvals = [-2*DS.prob.p[1]*t[1] for t in traj]

    ws2 = orthonormal(2, 2)
    ws22 = deepcopy(ws2)
    ws1 = SMatrix{2,2}(ws2)
    ws11 = deepcopy(ws1)

    te = DS.tangent
    te2 = DS2.tangent
    te_nojac = DS_nojac.tangent
    te2_nojac = DS2_nojac.tangent

    ws1 = evolve!(ws1, te)
    ws = evolve!(ws2, te2)
    ws11 = evolve!(ws11, te_nojac)
    ws22 = evolve!(ws22, te2_nojac)

    @test te.state == te2.state
    @test te_nojac.state == te2_nojac.state
    @test te.state == te2_nojac.state

    @test ws1 == ws2
    @test ws11 ≈ ws1
    @test ws22 ≈ ws1
    @test ws11 == ws22
    @test ws11 ≈ ws2

    ws1 = evolve!(ws1, te, 2)
    ws = evolve!(ws2, te2, 2)
    ws11 = evolve!(ws11, te_nojac, 2)
    ws22 = evolve!(ws22, te2_nojac, 2)

    @test te.state == te2.state
    @test te_nojac.state == te2_nojac.state
    @test te.state == te2_nojac.state

    @test ws1 == ws2
    @test ws11 ≈ ws1
    @test ws22 ≈ ws1
    @test ws11 == ws22
    @test ws11 ≈ ws2

    s = evolve(DS, 3)

    @test state(te) == s
    @test state(te2_nojac) == s
end


DS = Systems.henon()
DS2 = Systems.henon_iip()

DS_nojac = DDS(SVector(0.0, 0.0), DS.prob.f; p = DS.prob.p)
DS2_nojac = DDS([0.0, 0.0], DS2.prob.f; p = DS2.prob.p)


# Add small lyapunov test here
D = dimension(DS)
T = eltype(DS)
N = 10000
@testset "lyapunovs $v" for v in ["oop", "oop nojac", "iip", "iip nojac"]
    λ = @SVector zeros(T, D)
    if v == "oop"
        te = DS.tangent
        K = @SMatrix eye(T, D)
    elseif v == "oop nojac"
        te = DS_nojac.tangent
        K = @SMatrix eye(T, D)
    elseif v == "iip"
        te = DS2.tangent
        K = eye(T, D)
    elseif v == "iip nojac"
        te = DS2_nojac.tangent
        K = eye(T, D)
    end

    for i in 1:N
        K = evolve!(K, te)
        # println("K = $K")
        Q, R = qr(K)
        λ += log.(abs.(diag(R)))
        if v[1:3] == "oop"
            K = Q
        elseif v[1:3] == "iip"
            K .= Q
        end
    end
    λ = λ/N
    @test 0.418 < λ[1] < 0.422
    @test -1.63 < λ[2] < -1.61
end
