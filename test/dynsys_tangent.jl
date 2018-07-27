using DynamicalSystemsBase
using StaticArrays, LinearAlgebra
using Test

using DynamicalSystemsBase: CDS, DDS, DS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac
using DiffEqBase

println("\nTesting tangent dynamics...")

let
u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = ones(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]


# minimalistic lyapunov
function lyapunov_iip(ds::DS, k)
    D = dimension(ds)
    tode = tangent_integrator(ds, orthonormal(D,k))
    λ = zeros(k)
    for t in 1:1000
        while tode.t < t
            step!(tode)
        end
        # println("K = $K")
        QR = qr(get_deviations(tode))
        Q, R = QR.Q, QR.R
        λ .+= log.(abs.(diag(R)))

        set_deviations!(tode, Matrix(Q))
        u_modified!(tode, true)
    end
    λ = λ/tode.t # woooorks
end
function lyapunov_oop(ds::DS, k)
    D = dimension(ds)
    tode = tangent_integrator(ds, orthonormal(D,k))
    λ = zeros(k)
    ws_idx = SVector{k, Int}(collect(2:k+1))
    for t in 1:1000
        while tode.t < t
            step!(tode)
        end
        # println("K = $K")
        QR = qr(get_deviations(tode))
        Q, R = QR.Q, QR.R
        λ .+= log.(abs.(diag(R)))

        set_deviations!(tode, Q)
        u_modified!(tode, true)
    end
    λ./tode.t # woooorks
end

for i in 1:8
    @testset "combination $i" begin
        sysindx = i < 5 ? 1 : 2
        if i < 5
            if isodd(i)
                ds = CDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = CDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        else
            if isodd(i)
                ds = DDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
            else
                ds = DDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
            end
        end

        IIP = isinplace(ds)
        if IIP
            λ = lyapunov_iip(ds, 2)
        else
            λ = lyapunov_oop(ds, 2)
        end

        if i < 5
            @test 0.8 < λ[1] < 0.9
        else
            @test 0.4 < λ[1] < 0.45
        end
    end
end
end
