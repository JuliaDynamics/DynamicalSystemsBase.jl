using DynamicalSystemsBase, DiffEqNoiseProcess, Test, StochasticDiffEq
using DynamicalSystemsBase: find_noise_type, idfunc!
f!(du, u, p, t) = du .= 1.01u # deterministic part
σ = 0.25 # noise strength
get_prob(sde) = sde.integ.sol.prob
IIP = true

# Additive Wiener noise
W = WienerProcess(0.0, zeros(2), zeros(2))
addit_autom_inv = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
types = addit_autom_inv.noise_type
@test issetequal(types, [:additive, :autonomous, :linear, :invertiable])

# Scalar Wiener noise
W = WienerProcess(0.0, 0.0, 0.0)
scalar_a = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
types = scalar_a.noise_type
@test issetequal(types, [:scalar, :additive, :autonomous, :linear, :invertiable])

# invertiable correlated Wiener noise
W = CorrelatedWienerProcess([1 0.3; 0.3 1], 0.0, zeros(2), zeros(2))
corr_a = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
types = corr_a.noise_type
@test issetequal(types, [:additive, :autonomous, :linear, :invertiable])

# invertiable correlated Wiener noise
g!(du, u, p, t) = (du .= [1 0.3; 0.3 1]; return nothing)
corr_alt = CoupledSDEs(
    f!, g!, zeros(2), nothing, σ, noise_rate_prototype = zeros(2, 2),
    diffeq = (alg = EM(), dt = 0.1))
types = corr_alt.noise_type
@test issetequal(types, [:additive, :autonomous, :linear, :invertiable])

# non-invertiable correlated Wiener noise
W = CorrelatedWienerProcess([1 2; 2 4], 0.0, zeros(2), zeros(2))
corr_noninv = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise = W)
types = corr_noninv.noise_type
@test issetequal(types, [:additive, :autonomous, :linear, :non_invertiable])

# non-invertiable correlated Wiener noise
g!(du, u, p, t) = (du .= [1 0.3 1; 0.3 1 1]; return nothing)
corr_alt = CoupledSDEs(
    f!, g!, zeros(2), nothing, σ, noise_rate_prototype = zeros(2, 3),
    diffeq = (alg = EM(), dt = 0.1))
types = corr_alt.noise_type
@test issetequal(types, [:additive, :autonomous, :linear, :non_invertiable])

# multiplicative linear Wiener noise
g!(du, u, p, t) = (du .= u; return nothing)
linear_multipli = CoupledSDEs(f!, g!, rand(2) ./ 10, (), σ)
types = linear_multipli.noise_type
@test issetequal(types, [:multiplicative, :autonomous, :linear])

# non-diagonal multiplicative linear Wiener noise
lin_multipli_alt = CoupledSDEs(
    f!, g!, rand(2), (), σ, noise_rate_prototype = zeros(2, 3),
    diffeq = (alg = EM(), dt = 0.1))
types = lin_multipli_alt.noise_type
@test issetequal(types, [:multiplicative, :autonomous, :linear])

# multiplicative nonlinear Wiener noise
g!(du, u, p, t) = (du .= u .^ 2; return nothing)
nonlinear_multiplicative = CoupledSDEs(f!, g!, rand(2), (), σ)
types = nonlinear_multiplicative.noise_type
@test issetequal(types, [:multiplicative, :autonomous, :non_linear])
