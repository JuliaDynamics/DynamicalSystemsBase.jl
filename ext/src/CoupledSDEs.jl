using LinearAlgebra: LinearAlgebra

###########################################################################################
# DiffEq options
###########################################################################################
# SOSRA is only applicable for additive  noise and must be adaptive
# default sciml tolerance is 1e-2
const DEFAULT_SDE_SOLVER = SOSRA()
const DEFAULT_STOCH_DIFFEQ_KWARGS = (abstol=1e-2, reltol=1e-2, dt=0.1)
const DEFAULT_STOCH_DIFFEQ = (alg=DEFAULT_SDE_SOLVER, DEFAULT_STOCH_DIFFEQ_KWARGS...)

# Function from user `@xlxs4`, see
# https://github.com/JuliaDynamics/jl/pull/153
function _decompose_into_sde_solver_and_remaining(diffeq)
    if haskey(diffeq, :alg)
        return (diffeq[:alg], _delete(diffeq, :alg))
    else
        return (DEFAULT_SDE_SOLVER, diffeq)
    end
end

###########################################################################################
# Constructor functions
###########################################################################################

function DynamicalSystemsBase.CoupledSDEs(
    f,
    u0,
    p=SciMLBase.NullParameters();
    g=nothing,
    noise_strength=1.0,
    covariance=nothing,
    t0=0.0,
    diffeq=DEFAULT_STOCH_DIFFEQ,
    noise_prototype=nothing,
    noise_process=nothing,
    seed=UInt64(0)
)
    IIP = isinplace(f, 4) # from SciMLBase
    if !isnothing(g)
        @assert IIP == isinplace(g, 4) "f and g must both be in-place or out-of-place"
    end

    noise_type, cov = find_noise_type(g, u0, p, t0, noise_process, covariance, noise_prototype, IIP)
    g, noise_prototype = construct_diffusion_function(
        g, cov, noise_prototype, noise_strength, length(u0), IIP)

    s = correct_state(Val{IIP}(), u0)
    T = eltype(s)
    prob = SDEProblem{IIP}(
        f,
        g,
        s,
        (T(t0), T(Inf)),
        p;
        noise_rate_prototype=noise_prototype,
        noise=noise_process,
        seed=seed
    )
    return CoupledSDEs(prob, diffeq, noise_type)
end

function DynamicalSystemsBase.CoupledSDEs(
    prob::SDEProblem, diffeq=DEFAULT_STOCH_DIFFEQ, noise_type=nothing; special_kwargs...
)
    if haskey(special_kwargs, :diffeq)
        throw(
            ArgumentError(
                "`diffeq` is given as positional argument when an SDEProblem is provided."
            ),
        )
    end
    IIP = isinplace(prob) # from SciMLBase
    D = length(prob.u0)
    P = typeof(prob.p)
    if prob.tspan === (nothing, nothing)
        # If the problem was made via MTK, it is possible to not have a default timespan.
        U = eltype(prob.u0)
        prob = SciMLBase.remake(prob; tspan=(U(0), U(Inf)))
    end
    if isnothing(noise_type)
        noise_type, _ = find_noise_type(prob, IIP)
    end

    solver, remaining = _decompose_into_sde_solver_and_remaining(diffeq)
    integ = __init(
        prob,
        solver;
        remaining...,
        # Integrators are used exclusively iteratively. There is no reason to save anything.
        save_start=false,
        save_end=false,
        save_everystep=false,
        # DynamicalSystems.jl operates on integrators and `step!` exclusively,
        # so there is no reason to limit the maximum time evolution
        maxiters=Inf
    )
    return CoupledSDEs{IIP,D,typeof(integ),P}(
        integ, deepcopy(prob.p), diffeq, noise_type
    )
end

"""
    CoupledSDEs(ds::CoupledODEs, g, p [, σ]; kwargs...)

Converts a [`CoupledODEs`
](https://juliadynamics.github.io/DynamicalSystems.jl/stable/tutorial/#CoupledODEs)
system into a [`CoupledSDEs`](@ref).
"""
function DynamicalSystemsBase.CoupledSDEs(
    ds::CoupledODEs,
    p; # the parameter is likely changed as the diffusion function g is added.
    g=nothing,
    noise_strength=1.0,
    covariance=nothing,
    diffeq=DEFAULT_STOCH_DIFFEQ,
    noise_prototype=nothing,
    noise_process=nothing,
    seed=UInt64(0)
)
    return CoupledSDEs(
        dynamic_rule(ds),
        current_state(ds),
        p;
        g=g,
        noise_strength=noise_strength,
        covariance=covariance,
        diffeq=diffeq,
        noise_prototype=noise_prototype,
        noise_process=noise_process,
        seed=seed
    )
end

"""
    CoupledODEs(ds::CoupledSDEs; kwargs...)

Converts a [`CoupledSDEs`](@ref) into [`CoupledODEs`](@ref).
"""
function DynamicalSystemsBase.CoupledODEs(
    sys::CoupledSDEs; diffeq=DEFAULT_DIFFEQ, t0=0.0)
    return CoupledODEs(
        dynamic_rule(sys), SVector{length(sys.integ.u)}(sys.integ.u), sys.p0;
        diffeq=diffeq, t0=t0
    )
end

# Pretty print
function DynamicalSystemsBase.additional_details(ds::CoupledSDEs)
    solver, remaining = _decompose_into_sde_solver_and_remaining(ds.diffeq)
    return [
        "SDE solver" => string(nameof(typeof(solver))),
        "SDE kwargs" => remaining,
        "Noise type" => ds.noise_type
    ]
end

###########################################################################################
# API - obtaining information from the system
###########################################################################################

SciMLBase.isinplace(::CoupledSDEs{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::CoupledSDEs{IIP,D}) where {IIP,D} = D
DynamicalSystemsBase.current_state(ds::CoupledSDEs) = current_state(ds.integ)
DynamicalSystemsBase.isdeterministic(ds::CoupledSDEs) = false

function DynamicalSystemsBase.dynamic_rule(ds::CoupledSDEs)
    # with remake it can happen that we have nested SDEFunctions
    # sciml integrator deals with this internally well
    f = ds.integ.f
    while hasfield(typeof(f), :f)
        f = f.f
    end
    return f
end

function DynamicalSystemsBase.set_state!(ds::CoupledSDEs, u::AbstractArray)
    (set_state!(ds.integ, u); ds)
end
SciMLBase.step!(ds::CoupledSDEs, args...) = (step!(ds.integ, args...); ds)

# TODO We have to check if for SDEIntegrators a different step ReturnCode is possible.
function DynamicalSystemsBase.successful_step(integ::AbstractSDEIntegrator)
    rcode = integ.sol.retcode
    return rcode == SciMLBase.ReturnCode.Default || rcode == SciMLBase.ReturnCode.Success
end

# This is here to ensure that `u_modified!` is called
function DynamicalSystemsBase.set_parameter!(ds::CoupledSDEs, args...)
    _set_parameter!(ds, args...)
    u_modified!(ds.integ, true)
    return nothing
end

DynamicalSystemsBase.referrenced_sciml_prob(ds::CoupledSDEs) = ds.integ.sol.prob

function covariance(ds::CoupledSDEs)
    A = diffusion_matrix(ds)
    A * A'
end

function diffusion_matrix(ds::CoupledSDEs{IIP,D}) where {IIP,D}
    if ds.noise_type[:invertible]
        diffusion = diffusion_function(ds)
        A = diffusion(zeros(D), current_parameters(ds), 0.0)
    else
        A = nothing
    end
    return A
end

###########################################################################################
# Utilities
###########################################################################################

"""
compute diffusion matrix given the covariance matrix of noise matrix.
"""
function diffusion_matrix(Γ::AbstractMatrix)
    U, S, V = LinearAlgebra.svd(Γ) # A ≈ U * Diagonal(S) * V'
    A = U * LinearAlgebra.Diagonal(sqrt.(S))
    return A
end

"""
https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#The-uniform-scaling-operator
"""
function construct_diffusion_function(g, cov, noise_prototype, σ, D, IIP)
    if isnothing(g) # diagonal additive noise
        cov = isnothing(cov) ? LinearAlgebra.I(D) : cov
        size(cov)[1] != D &&
            throw(ArgumentError("covariance matrix must be of size $((D, D))"))
        A = diffusion_matrix(cov)
        if IIP
            if isdiag(cov)
                g = (du, u, p, t) -> du .= σ .* diag(A)
            else
                g = (du, u, p, t) -> du .= σ .* A
                noise_prototype = zeros(size(A))
                # ^ we could make this sparse to make it more performant
            end
        else
            if isdiag(cov)
                g = (u, p, t) -> SVector{length(diag(A)),eltype(A)}(diag(σ .* A))
            else
                g = (u, p, t) -> SMatrix{size(A)...,eltype(A)}(σ .* A)
                noise_prototype = zeros(size(A))
            end
        end
    end
    return g, noise_prototype
end
