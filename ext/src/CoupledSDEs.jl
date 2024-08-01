###########################################################################################
# DiffEq options
###########################################################################################

const DEFAULT_SDE_SOLVER = SOSRI() # default sciml solver
const DEFAULT_STOCH_DIFFEQ_KWARGS = (abstol = 1e-2, reltol = 1e-2) # default sciml tol
const DEFAULT_STOCH_DIFFEQ = (alg = DEFAULT_SDE_SOLVER, DEFAULT_STOCH_DIFFEQ_KWARGS...)

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
        g,
        u0,
        p = SciMLBase.NullParameters(),
        noise_strength = 1.0;
        t0 = 0.0,
        diffeq = DEFAULT_STOCH_DIFFEQ,
        noise_rate_prototype = nothing,
        noise = nothing,
        seed = UInt64(0)
)
    IIP = isinplace(f, 4) # from SciMLBase
    @assert IIP==isinplace(g, 4) "f and g must both be in-place or out-of-place"

    s = correct_state(Val{IIP}(), u0)
    T = eltype(s)
    prob = SDEProblem{IIP}(
        f,
        g,
        s,
        (T(t0), T(Inf)),
        p;
        noise_rate_prototype = noise_rate_prototype,
        noise = noise,
        seed = seed
    )
    return CoupledSDEs(prob, diffeq, noise_strength)
end

function DynamicalSystemsBase.CoupledSDEs(
        prob::SDEProblem, diffeq = DEFAULT_STOCH_DIFFEQ, noise_strength = 1.0; special_kwargs...
)
    if haskey(special_kwargs, :diffeq)
        throw(
            ArgumentError(
            "`diffeq` is given as positional argument when an ODEProblem is provided."
        ),
        )
    end
    IIP = isinplace(prob) # from SciMLBase
    D = length(prob.u0)
    P = typeof(prob.p)
    if prob.tspan === (nothing, nothing)
        # If the problem was made via MTK, it is possible to not have a default timespan.
        U = eltype(prob.u0)
        prob = SciMLBase.remake(prob; tspan = (U(0), U(Inf)))
    end
    sde_function = SDEFunction(prob.f, add_noise_strength(noise_strength, prob.g, IIP))
    prob = SciMLBase.remake(prob; f = sde_function)
    noise_type = find_noise_type(prob, IIP)

    solver, remaining = _decompose_into_sde_solver_and_remaining(diffeq)
    integ = __init(
        prob,
        solver;
        remaining...,
        # Integrators are used exclusively iteratively. There is no reason to save anything.
        save_start = false,
        save_end = false,
        save_everystep = false,
        # DynamicalSystems.jl operates on integrators and `step!` exclusively,
        # so there is no reason to limit the maximum time evolution
        maxiters = Inf
    )
    return CoupledSDEs{IIP, D, typeof(integ), P, true}(
        integ, deepcopy(prob.p), noise_strength, diffeq, noise_type
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
        g,
        p, # the parameter is likely changed as the diffusion function g is added.
        noise_strength = 1.0;
        diffeq = DEFAULT_STOCH_DIFFEQ,
        noise_rate_prototype = nothing,
        noise = nothing,
        seed = UInt64(0)
)
    return CoupledSDEs(
        dynamic_rule(ds),
        g,
        current_state(ds),
        p,
        noise_strength;
        diffeq = diffeq,
        noise_rate_prototype = noise_rate_prototype,
        noise = noise,
        seed = seed
    )
end

"""
    CoupledODEs(ds::CoupledSDEs; kwargs...)

Converts a [`CoupledSDEs`](@ref) into [`CoupledODEs`](@ref).
"""
function DynamicalSystemsBase.CoupledODEs(
        sys::CoupledSDEs; diffeq = DEFAULT_DIFFEQ, t0 = 0.0)
    return CoupledODEs(
        dynamic_rule(sys), SVector{length(sys.integ.u)}(sys.integ.u), sys.p0;
        diffeq = diffeq, t0 = t0
    )
end

# Pretty print
function DynamicalSystemsBase.additional_details(ds::CoupledSDEs)
    solver, remaining = _decompose_into_sde_solver_and_remaining(ds.diffeq)
    return [
        "Noise strength" => ds.noise_strength,
        "SDE solver" => string(nameof(typeof(solver))),
        "SDE kwargs" => remaining,
        "Noise type" => ds.noise_type
    ]
end

###########################################################################################
# API - obtaining information from the system
###########################################################################################

SciMLBase.isinplace(::CoupledSDEs{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::CoupledSDEs{IIP, D}) where {IIP, D} = D
DynamicalSystemsBase.current_state(ds::CoupledSDEs) = current_state(ds.integ)
DynamicalSystemsBase.isdeterministic(ds::CoupledSDEs) = false

# so that `ds` is printed
function DynamicalSystemsBase.set_state!(ds::CoupledSDEs, u::AbstractArray)
    (set_state!(ds.integ, u); ds)
end
SciMLBase.step!(ds::CoupledSDEs, args...) = (step!(ds.integ, args...); ds)

# TODO We have to check if for SDEIntegrators a different step interruption is possible.
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

###########################################################################################
# Utilities
###########################################################################################

function σg(σ, g)
    return (u, p, t) -> σ .* g(u, p, t)
end

function σg!(σ, g!)
    function (du, u, p, t)
        g!(du, u, p, t)
        du .*= σ
        return nothing
    end
end

function add_noise_strength(σ, g, IIP)
    newg = IIP ? σg!(σ, g) : σg(σ, g)
    return newg
end
