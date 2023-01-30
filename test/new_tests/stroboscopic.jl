using DynamicalSystemsBase, Test

using OrdinaryDiffEq: Vern9

include("test_system_function.jl")

@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
end
function duffing_rule_iip(du, u, p, t)
    du .= duffing_rule(u, p, t)
    return nothing
end

u0 = [0.1, 0.25]
p0 = [1.0, 0.3, 0.2, -1]
T = 2π/1.0

duffing_oop = StroboscopicMap(CoupledODEs(duffing_rule, u0, p0), T)
duffing_iip = StroboscopicMap(T, duffing_rule_iip, copy(u0), p0)
duffing_vern = StroboscopicMap(T, duffing_rule, u0, p0;
    diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)
)

for (ds, iip) in zip((duffing_oop, duffing_iip, duffing_vern), (false, true, false))

    @test dynamic_rule(ds) == (iip ? duffing_rule_iip : duffing_rule)
    name = (ds === duffing_vern) ? "duffvern" : "duffing"
    test_dynamical_system(ds, u0, name, false, iip)

end

# TODO: Add extra test that duffing goes to fixed point
# with
# ω = 2.2, f = 27.0, d = 0.2, β = 1

# TODO: reproduce this video, to ensure `reinit!` works as expected:
# https://juliadynamics.github.io/DynamicalSystems.jl/dev/chaos/orbitdiagram/#Stroboscopic-Map
# using `reinit!`