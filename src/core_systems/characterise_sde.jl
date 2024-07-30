using StochasticDiffEq: WienerProcess, CorrelatedWienerProcess, EM, SDEProblem
using LinearAlgebra

function is_state_independent(g, u, p, t)
    rdm_states = [rand(eltype(u), length(u)) for _ in 1:10]
    val = map(u -> g(u, p, t), rdm_states)
    length(unique(val)) == 1
end
function is_time_independent(g, u, p)
    trange = 0:0.1:1
    val = map(t -> g(u, p, t), trange) |> unique |> length == 1
    length(unique(val)) == 1
end
is_invertible(x) = issuccess(lu(x, check = false))

function is_linear(f, x, y, c)
    check1 = f(x + y) == f(x) + f(y)
    check2 = f(c * x) == c * f(x)
    return check1 && check2
end

function find_noise_type(prob::SDEProblem, IIP)
    noise = prob.noise
    noise_rate_prototype = prob.noise_rate_prototype
    noise_rate_size = isnothing(noise_rate_prototype) ? nothing : size(noise_rate_prototype)
    covariance = isnothing(noise) ? nothing : noise.covariance
    param = prob.p
    u0 = prob.u0
    D = length(u0)

    isadditive = false
    isautonomous = false
    islinear = false
    isinvertible = false

    function g(u, p, t)
        if IIP
            du = deepcopy(isnothing(noise_rate_prototype) ? u : noise_rate_prototype)
            prob.g(du, u, p, t)
            return du
        else
            return prob.g(u, p, t)
        end
    end
    time_independent = is_time_independent(g, rand(D), param)
    state_independent = is_state_independent(g, u0, param, 1.0)

    # additive noise is equal to state independent noise
    isadditive = state_independent
    isautonomous = time_independent
    islinear = !state_independent ?
               is_linear(u -> g(u, param, 1.0), rand(D), rand(D), 2.0) : true

    if time_independent && state_independent
        if noise_rate_size == (D, D) && !isnothing(covariance)
            error("The diffusion function `g` acts as an covariance matrix but the noise process W also has a covariance matrix. This is ambiguous.")
        elseif noise_rate_size == (D, D) && isnothing(covariance)
            covariance = g(zeros(D), param, 0.0)
            isinvertible = is_invertible(covariance)
        elseif !isnothing(noise_rate_size) && noise_rate_size != (D, D)
            isinvertible = false
        else
            isinvertible = isnothing(covariance) || is_invertible(covariance)
        end
    end

    return (additive = isadditive, autonomous = isautonomous,
        linear = islinear, invertible = isinvertible)
end # function
