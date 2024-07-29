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
# isinvertible(x) = applicable(inv, x) && isone(inv(x) * x)
isinvertible(x) = issuccess(lu(x, check = false))

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

    properties = Symbol[]

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
    push!(properties, state_independent ? :additive : :multiplicative)
    push!(properties, time_independent ? :autonomous : :non_autonomous)
    if !state_independent
        linear = is_linear(u -> g(u, param, 1.0), rand(D), rand(D), 2.0)
        push!(properties, linear ? :linear : :non_linear)
    else
        push!(properties, :linear)
    end

    if time_independent && state_independent
        if noise_rate_size == (D, D) && !isnothing(covariance)
            error(
                "The diffusion function `g` acts as an covariance matrix but
                the noise process W also has a covariance matrix. This is ambiguous.")
        elseif noise_rate_size == (D, D) && isnothing(covariance)
            covariance = g(zeros(D), param, 0.0)
            if !isinvertible(covariance)
                push!(properties, :non_invertible)
            else
                push!(properties, :invertible)
            end
        elseif !isnothing(covariance) && !isinvertible(covariance)
            push!(properties, :non_invertible)
        elseif !isnothing(noise_rate_size) && noise_rate_size != (D, D)
            push!(properties, :non_invertible)
        else
            push!(properties, :invertible)
        end
    end

    if !isnothing(noise) && D != length(noise.dW) && length(noise.dW) == 1
        push!(properties, :scalar)
    end
    return properties
end # function
