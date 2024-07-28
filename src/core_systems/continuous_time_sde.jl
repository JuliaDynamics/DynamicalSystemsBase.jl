using SciMLBase: SDEProblem, AbstractSDEIntegrator, __init, SDEFunction
using StochasticDiffEq: SOSRA
export CoupledSDEs

###########################################################################################
# DiffEq options
###########################################################################################
const DEFAULT_SDE_SOLVER = SOSRA() # default recommendation for additive noise
const DEFAULT_STOCH_DIFFEQ_KWARGS = (abstol = 1e-2, reltol = 1e-2)
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
# Type
###########################################################################################
"""
    CoupledSDEs <: ContinuousTimeDynamicalSystem
    CoupledSDEs(f, g, u0 [, p, σ]; kwargs...)

A stochastic continuous time dynamical system defined by a set of
coupled ordinary differential equations as follows:
```math
d\\vec{u} = \\vec{f}(\\vec{u}, p, t) dt + σ \\vec{g}(\\vec{u}, p, t) dW_t
```

Optionally provide the overall noise strength `σ`, the parameter container `p`
and initial time as keyword `t0`. By defualt `σ=1` and `t0=0`.

For construction instructions regarding `f, u0` see the [DynamicalSystems.jl tutorial
](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/dev/tutorial/).

The stochastic part of the differential equation is defined by the function `g` and
the keyword arguments `noise_rate_prototype` and `noise`. `noise` indicates
the noise process applied and defaults to Gaussian white noise (Wiener process).
For details on defining various noise processes, refer to [DiffEqNoiseProcess.jl
](https://docs.sciml.ai/DiffEqNoiseProcess/stable/).
`noise_rate_prototype` indicates the prototype type instance for the noise rates, i.e.,
the output of `g`. It can be any type which overloads `A_mul_B!` with itself being the
middle argument. Commonly, this is a matrix or sparse matrix. If this is not given, it
defaults to `nothing`, which means the `g` should be interpreted as being diagonal.

## DifferentialEquations.jl interfacing

The ODEs are evolved via the solvers of DifferentialEquations.jl.
When initializing a `CoupledODEs`, you can specify the solver that will integrate
`f` in time, along with any other integration options, using the `diffeq` keyword.
For example you could use `diffeq = (abstol = 1e-9, reltol = 1e-9)`.
If you want to specify a solver, do so by using the keyword `alg`, e.g.:
`diffeq = (alg = Tsit5(), reltol = 1e-6)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers. The default `diffeq` is:

```julia
$(DEFAULT_STOCH_DIFFEQ)
```

`diffeq` keywords can also include `callback` for [event handling
](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

Dev note: `CoupledSDEs` is a light wrapper of  `SDEIntegrator` from StochasticDiffEq.jl.
The integrator is available as the field `integ`, and the `SDEProblem` is `integ.sol.prob`.
The convenience syntax `SDEProblem(ds::CoupledSDEs, tspan = (t0, Inf))` is available
to extract the problem.
"""
struct CoupledSDEs{IIP,D,I,P,S} <: ContinuousTimeDynamicalSystem
    # D parametrised by length of u0
    # S indicated if the noise strength has been added to the diffusion function
    integ::I
    # things we can't recover from `integ`
    p0::P
    noise_strength
    diffeq # isn't parameterized because it is only used for display
end

function CoupledSDEs(
    f,
    g,
    u0,
    p=SciMLBase.NullParameters(),
    noise_strength=1.0;
    t0=0.0,
    diffeq=DEFAULT_STOCH_DIFFEQ,
    noise_rate_prototype=nothing,
    noise=nothing,
    seed=UInt64(0),
)
    IIP = isinplace(f, 4) # from SciMLBase
    @assert IIP == isinplace(g, 4) "f and g must both be in-place or out-of-place"

    s = correct_state(Val{IIP}(), u0)
    T = eltype(s)
    prob = SDEProblem{IIP}(
        f,
        g,
        s,
        (T(t0), T(Inf)),
        p;
        noise_rate_prototype=noise_rate_prototype,
        noise=noise,
        seed=seed,
    )
    return CoupledSDEs(prob, diffeq, noise_strength)
end

function CoupledSDEs(
    prob::SDEProblem, diffeq=DEFAULT_STOCH_DIFFEQ, noise_strength=1.0; special_kwargs...
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
        prob = SciMLBase.remake(prob; tspan=(U(0), U(Inf)))
    end
    sde_function = SDEFunction(prob.f, add_noise_strength(noise_strength, prob.g, IIP))
    prob = SciMLBase.remake(prob; f=sde_function)

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
        maxiters=Inf,
    )
    return CoupledSDEs{IIP,D,typeof(integ),P,true}(
        integ, deepcopy(prob.p), noise_strength, diffeq
    )
end

"""
    CoupledSDEs(ds::CoupledODEs, g, p [, σ]; kwargs...)

Converts a [`CoupledODEs`
](https://juliadynamics.github.io/DynamicalSystems.jl/stable/tutorial/#CoupledODEs)
system into a [`CoupledSDEs`](@ref).
"""
function CoupledSDEs(
    ds::CoupledODEs,
    g,
    p, # the parameter is likely changed as the diffusion function g is added.
    noise_strength=1.0;
    diffeq=DEFAULT_STOCH_DIFFEQ,
    noise_rate_prototype=nothing,
    noise=nothing,
    seed=UInt64(0),
)
    return CoupledSDEs(
        dynamic_rule(ds),
        g,
        current_state(ds),
        p,
        noise_strength;
        diffeq=diffeq,
        noise_rate_prototype=noise_rate_prototype,
        noise=noise,
        seed=seed,
    )
end

"""
    CoupledODEs(ds::CoupledSDEs; kwargs...)

Converts a [`CoupledSDEs`](@ref) into [`CoupledODEs`](@ref).
"""
function CoupledODEs(sys::CoupledSDEs; diffeq=DEFAULT_DIFFEQ, t0=0.0)
    return CoupledODEs(
        sys.integ.f, SVector{length(sys.integ.u)}(sys.integ.u), sys.p0; diffeq=diffeq, t0=t0
    )
end

# Pretty print
function additional_details(ds::CoupledSDEs)
    solver, remaining = _decompose_into_sde_solver_and_remaining(ds.diffeq)
    return [
        "Noise strength" => ds.noise_strength,
        "SDE solver" => string(nameof(typeof(solver))),
        "SDE kwargs" => remaining,
    ]
end

###########################################################################################
# API - obtaining information from the system
###########################################################################################

SciMLBase.isinplace(::CoupledSDEs{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::CoupledSDEs{IIP,D}) where {IIP,D} = D
current_state(ds::CoupledSDEs) = current_state(ds.integ)
isdeterministic(ds::CoupledSDEs) = false

function dynamic_rule(sys::CoupledSDEs)
    f = sys.integ.f
    while hasfield(typeof(f), :f)
        f = f.f
    end
    return f
end

# so that `ds` is printed
set_state!(ds::CoupledSDEs, u::AbstractArray) = (set_state!(ds.integ, u); ds)
SciMLBase.step!(ds::CoupledSDEs, args...) = (step!(ds.integ, args...); ds)

function successful_step(integ::AbstractSDEIntegrator)
    rcode = integ.sol.retcode
    return rcode == SciMLBase.ReturnCode.Default || rcode == SciMLBase.ReturnCode.Success
end

# This is here to ensure that `u_modified!` is called
function set_parameter!(ds::CoupledSDEs, args...)
    _set_parameter!(ds, args...)
    u_modified!(ds.integ, true)
    return nothing
end

referrenced_sciml_prob(ds::CoupledSDEs) = ds.integ.sol.prob

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

"""
    idfunc(u, p, t)

Identity function for a diffusion function `g` of `CoupledSDEs` (out-of-place).
Equivalent to `(u, p, t) -> ones(length(u))`,
"""
function idfunc(u, p, t)
    return  SVector{length(u)}(ones(eltype(u), length(u)))
end;

"""
    idfunc!(du, u, p, t)

Identity function for a diffusion function `g` of `CoupledSDEs` (in-place).
Equivalent to `idfunc!(du, u, p, t) = (du .= ones(length(u)); return nothing)`
"""
function idfunc!(du, u, p, t)
    du .= ones(eltype(u), length(u))
    return nothing
end;
