using DynamicalSystemsBase, DiffEqNoiseProcess, Test, StochasticDiffEq
using DynamicalSystemsBase: find_noise_type, idfunc!
f!(du, u, p, t) = du .= 1.01u # deterministic part
σ = 0.25 # noise strength
get_prob(sde) = sde.integ.sol.prob
IIP = true

# function idfunc(u, p, t)
#     return  SVector{length(u)}(ones(eltype(u), length(u)))
# end;

function idfunc!(du, u, p, t)
    du .= ones(eltype(u), length(u))
    return nothing
end;

@testset "Additive" begin
    # Additive Wiener noise
    W = WienerProcess(0.0, zeros(2), zeros(2))
    addit_autom_inv = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
    types = addit_autom_inv.noise_type
    @test values(types) == (true, true, true, true)

    # Scalar Wiener noise
    W = WienerProcess(0.0, 0.0, 0.0)
    scalar_a = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
    types = scalar_a.noise_type
    @test values(types) == (true, true, true, true)

    # invertible correlated Wiener noise
    W = CorrelatedWienerProcess([1 0.3; 0.3 1], 0.0, zeros(2), zeros(2))
    corr_a = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
    types = corr_a.noise_type
    @test values(types) == (true, true, true, true)

    # invertible correlated Wiener noise
    g!(du, u, p, t) = (du .= [1 0.3; 0.3 1]; return nothing)
    corr_alt = CoupledSDEs(
        f!, g!, zeros(2), nothing, σ, noise_rate_prototype = zeros(2, 2),
        diffeq = (alg = EM(), dt = 0.1))
    types = corr_alt.noise_type
    @test values(types) == (true, true, true, true)

    # non-invertible correlated Wiener noise
    W = CorrelatedWienerProcess([1 2; 2 4], 0.0, zeros(2), zeros(2))
    corr_noninv = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
    types = corr_noninv.noise_type
    @test values(types) == (true, true, true, false)

    # non-invertible correlated Wiener noise
    g!(du, u, p, t) = (du .= [1 0.3 1; 0.3 1 1]; return nothing)
    corr_alt = CoupledSDEs(
        f!, g!, zeros(2), nothing, σ, noise_rate_prototype = zeros(2, 3),
        diffeq = (alg = EM(), dt = 0.1))
    types = corr_alt.noise_type
    @test values(types) == (true, true, true, false)

    # non-autonomous Wiener noise
    g!(du, u, p, t) = (du .= 1 / (1 + t); return nothing)
    W = WienerProcess(0.0, zeros(2), zeros(2))
    addit_non_autom = CoupledSDEs(f!, g!, zeros(2), nothing, σ; noise = W)
    types = addit_non_autom.noise_type
    @test values(types) == (true, false, true, false)
end

@testset "Multiplicative" begin
    # multiplicative linear Wiener noise
    g!(du, u, p, t) = (du .= u; return nothing)
    linear_multipli = CoupledSDEs(f!, g!, rand(2) ./ 10, (), σ)
    types = linear_multipli.noise_type
    @test values(types) == (false, true, true, false)

    # non-diagonal multiplicative linear Wiener noise
    lin_multipli_alt = CoupledSDEs(
        f!, g!, rand(2), (), σ, noise_rate_prototype = zeros(2, 3),
        diffeq = (alg = EM(), dt = 0.1))
    types = lin_multipli_alt.noise_type
    @test values(types) == (false, true, true, false)

    # multiplicative nonlinear Wiener noise
    g!(du, u, p, t) = (du .= u .^ 2; return nothing)
    nonlinear_multiplicative = CoupledSDEs(f!, g!, rand(2), (), σ)
    types = nonlinear_multiplicative.noise_type
    @test values(types) == (false, true, false, false)

    # non-autonomous linear multiplicative Wiener noise
    g!(du, u, p, t) = (du .= u ./ (1 + t); return nothing)
    W = WienerProcess(0.0, zeros(2), zeros(2))
    addit_non_autom = CoupledSDEs(f!, g!, zeros(2), nothing, σ; noise = W)
    types = addit_non_autom.noise_type
    @test values(types) == (false, false, true, false)

    # non-autonomous nonlinear multiplicative Wiener noise
    g!(du, u, p, t) = (du .= u.^2 ./ (1 + t); return nothing)
    W = WienerProcess(0.0, zeros(2), zeros(2))
    addit_non_autom = CoupledSDEs(f!, g!, zeros(2), nothing, σ; noise = W)
    types = addit_non_autom.noise_type
    @test values(types) == (false, false, false, false)
end
