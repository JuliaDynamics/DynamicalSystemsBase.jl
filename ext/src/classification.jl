using LinearAlgebra

function is_state_independent(g, u, p, t)
    rdm_states = [rand(eltype(u), length(u)) for _ in 1:10]
    val = map(u -> g(u, p, t), rdm_states)
    length(unique(val)) == 1
end
function is_time_independent(g, u, p, t0)
    trange = t0:0.1:10
    val = map(t -> g(u, p, t), trange)
    length(unique(val)) == 1
end
function is_invertible(x; tol=1e-10)
    F = lu(x, check=false)
    det = abs(prod(diag(F.U)))
    return det > tol
end

function is_linear(f, x, y, c)
    check1 = f(x + y) == f(x) + f(y)
    check2 = f(c * x) == c * f(x)
    return check1 && check2
end

function diffusion_function(g, IIP, noise_prototype)
    function diffusion(u, p, t)
        if IIP
            du = deepcopy(isnothing(noise_prototype) ? u : noise_prototype)
            g(du, u, p, t)
            return du
        else
            return g(u, p, t)
        end
    end
end
function diffusion_function(ds::CoupledSDEs{IIP}) where {IIP}
    diffusion_function(ds.integ.g, IIP, referrenced_sciml_prob(ds).noise_rate_prototype)
end

"""
We classify the noise type of the CoupledSDEs based on the system given by the user.
In doing this we also determine the covariance matrix

"""
function find_noise_type(g, u0, p, t0, noise, covariance, noise_prototype, IIP)
    noise_size = isnothing(noise_prototype) ? nothing : size(noise_prototype)
    noise_cov = isnothing(noise) ? nothing : noise.covariance
    D = length(u0)

    if !isnothing(noise_cov)
        throw(
            ArgumentError("CoupledSDEs does not support correlation between noise processes through DiffEqNoiseProcess.jl interface. Instead, use the `covariance` kwarg of `CoupledSDEs`.")
        )
    end

    isadditive = false
    isautonomous = false
    islinear = false
    isinvertible = false

    diffusion = diffusion_function(g, IIP, noise_prototype)

    if isnothing(g)
        isadditive = true
        isautonomous = true
        islinear = true
        if isnothing(covariance)
            covariance = LinearAlgebra.I(D)
            isinvertible = true
        else
            isinvertible = is_invertible(covariance)
        end
    elseif !isnothing(covariance)
        throw(
            ArgumentError("Both `g` and `covariance` are provided. Instead opt to encode the covariance in the difussion function `g` with the `noise_prototype` kwarg.")
        )
    else
        time_independent = is_time_independent(diffusion, rand(D), p, t0)
        state_independent = is_state_independent(diffusion, u0, p, t0)

        # additive noise is equal to state independent noise
        isadditive = state_independent
        isautonomous = time_independent
        islinear = !state_independent ?
                   is_linear(u -> diffusion(u, p, t0), rand(D), rand(D), 2.0) : true

        if time_independent && state_independent
            if !isnothing(noise_size) && isequal(noise_size...)
                A = diffusion(zeros(D), p, 0.0)
                covariance = A * A'
                isinvertible = is_invertible(covariance)
            elseif !isnothing(noise_size) && !isequal(noise_size...)
                isinvertible = false
                covariance = nothing
            else
                isinvertible = true
                covariance = LinearAlgebra.I(D)
            end
        else
            covariance = nothing
        end
    end

    noise_type = (additive=isadditive, autonomous=isautonomous,
        linear=islinear, invertible=isinvertible)
    return noise_type, covariance
end

function find_noise_type(prob::SDEProblem, IIP)
    find_noise_type(
        prob.g, prob.u0, prob.p, prob.tspan[1], prob.noise,
        nothing, prob.noise_rate_prototype, IIP)
end
