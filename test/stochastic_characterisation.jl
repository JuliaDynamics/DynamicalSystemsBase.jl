using DynamicalSystemsBase, DiffEqNoiseProcess, Test, StochasticDiffEq

sciml_prob(sde) = sde.integ.sol.prob
f!(du, u, p, t) = du .= 1.01u

@testset "Additive noise" begin
    # Additive Wiener noise
    W = WienerProcess(0.0, zeros(2), zeros(2))
    addit_autom_inv = CoupledSDEs(f!, zeros(2); noise_process = W)
    types = addit_autom_inv.noise_type
    @test values(types) == (true, true, true, true)
    @test CoupledSDEs(sciml_prob(addit_autom_inv)).noise_type == types

    # Scalar Wiener noise
    W = WienerProcess(0.0, 0.0, 0.0)
    scalar_a = CoupledSDEs(f!, zeros(2); noise_process = W)
    types = scalar_a.noise_type
    @test values(types) == (true, true, true, true)
    @test CoupledSDEs(sciml_prob(scalar_a)).noise_type == types

    # invertible correlated Wiener noise
    corr = CoupledSDEs(f!, zeros(2); covariance = [1 0.3; 0.3 1])
    types = corr.noise_type
    @test values(types) == (true, true, true, true)
    @test CoupledSDEs(sciml_prob(corr)).noise_type == types

    # invertible correlated Wiener noise
    g!(du, u, p, t) = (du .= [1 0.3; 0.3 1]; return nothing)
    corr_alt = CoupledSDEs(f!, zeros(2); g = g!, noise_prototype = zeros(2, 2))
    types = corr_alt.noise_type
    @test values(types) == (true, true, true, true)
    @test CoupledSDEs(sciml_prob(corr_alt)).noise_type == types

    # non-invertible correlated Wiener noise
    corr_noninv = CoupledSDEs(f!, zeros(2); covariance = [1 2; 2 4])
    types = corr_noninv.noise_type
    @test values(types) == (true, true, true, false)
    @test CoupledSDEs(sciml_prob(corr_noninv)).noise_type == types

    # non-invertible correlated Wiener noise
    g!(du, u, p, t) = (du .= [1 0.3 1; 0.3 1 1]; return nothing)
    corr_alt = CoupledSDEs(f!, zeros(2); g = g!, noise_prototype = zeros(2, 3))
    types = corr_alt.noise_type
    @test values(types) == (true, true, true, false)
    @test CoupledSDEs(sciml_prob(corr_alt)).noise_type == types

    # non-autonomous Wiener noise
    g!(du, u, p, t) = (du .= 1 / (1 + t); return nothing)
    addit_non_autom = CoupledSDEs(f!, zeros(2); g = g!)
    types = addit_non_autom.noise_type
    @test values(types) == (true, false, true, false)
    @test CoupledSDEs(sciml_prob(addit_non_autom)).noise_type == types
end

@testset "Multiplicative noise" begin
    # multiplicative linear Wiener noise
    g!(du, u, p, t) = (du .= u; return nothing)
    linear_multipli = CoupledSDEs(f!, rand(2) ./ 10; g = g!)
    types = linear_multipli.noise_type
    @test values(types) == (false, true, true, false)
    @test CoupledSDEs(sciml_prob(linear_multipli)).noise_type == types

    # non-diagonal multiplicative linear Wiener noise
    lin_multipli_alt = CoupledSDEs(f!, rand(2); g = g!, noise_prototype = zeros(2, 3))
    types = lin_multipli_alt.noise_type
    @test values(types) == (false, true, true, false)
    @test CoupledSDEs(sciml_prob(lin_multipli_alt)).noise_type == types

    # multiplicative nonlinear Wiener noise
    g!(du, u, p, t) = (du .= u .^ 2; return nothing)
    nonlinear_multiplicative = CoupledSDEs(f!, rand(2); g = g!)
    types = nonlinear_multiplicative.noise_type
    @test values(types) == (false, true, false, false)
    @test CoupledSDEs(sciml_prob(nonlinear_multiplicative)).noise_type == types

    # non-autonomous linear multiplicative Wiener noise
    g!(du, u, p, t) = (du .= u ./ (1 + t); return nothing)
    addit_non_autom = CoupledSDEs(f!, zeros(2); g = g!)
    types = addit_non_autom.noise_type
    @test values(types) == (false, false, true, false)
    @test CoupledSDEs(sciml_prob(addit_non_autom)).noise_type == types

    # non-autonomous nonlinear multiplicative Wiener noise
    g!(du, u, p, t) = (du .= u .^ 2 ./ (1 + t); return nothing)
    addit_non_autom = CoupledSDEs(f!, zeros(2); g = g!)
    types = addit_non_autom.noise_type
    @test values(types) == (false, false, false, false)
    @test CoupledSDEs(sciml_prob(addit_non_autom)).noise_type == types
end
