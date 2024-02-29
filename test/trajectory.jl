using DynamicalSystemsBase
using Test

@testset "trajectory with `nothing`" begin

    # time-dependent logistic map, so that the `r` parameter increases with time
    r1 = 3.83
    r2 = 3.86
    N = 2000
    rs = range(r1, r2; length = N)

    function logistic_drifting_rule(u, rs, n)
        r = rs[n+1] # time is `n`, starting from 0
        return SVector(r*u[1]*(1 - u[1]))
    end

    ds = DeterministicIteratedMap(logistic_drifting_rule, [0.5], rs)

    x, t = trajectory(ds, N-1)
    @test length(x) > 1
end
