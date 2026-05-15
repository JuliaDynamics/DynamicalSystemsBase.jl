export CoupledSDEs

using SciMLBase: SDEProblem, AbstractSDEIntegrator, SDEFunction, __init
using StochasticDiffEqHighOrder: SOSRA
using StochasticDiffEqHighOrder.StochasticDiffEqCore: setup_next_step!
using LinearAlgebra
import Random

"""
    CoupledSDEs <: ContinuousTimeDynamicalSystem
    CoupledSDEs(f, u0 [, p]; kwargs...)

A stochastic continuous time dynamical system defined by a set of
coupled stochastic differential equations (SDEs) as follows:
```math
\\text{d}\\mathbf{u} = \\mathbf{f}(\\mathbf{u}, p, t) \\text{d}t + \\mathbf{g}(\\mathbf{u}, p, t) \\text{d}\\mathcal{N}_t
```
where
``\\mathbf{u}(t)`` is the state vector at time ``t``, ``\\mathbf{f}`` describes the
deterministic dynamics, and the noise term
``\\mathbf{g}(\\mathbf{u}, p, t) \\text{d}\\mathcal{N}_t`` describes
the stochastic forcing in terms of a noise function (or *diffusion function*)
``\\mathbf{g}`` and a noise process ``\\mathcal{N}_t``. The parameters of the functions
``\\mathcal{f}`` and ``\\mathcal{g}`` are contained in the vector ``p``.

There are multiple ways to construct a `CoupledSDEs` depending on the type of stochastic
forcing.

The only required positional arguments are the deterministic dynamic rule
`f(u, p, t)`, the initial state `u0`, and optinally the parameter container `p`
(by default `p = nothing`). For construction instructions regarding `f, u0` see the [DynamicalSystems.jl tutorial
](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/dev/tutorial/).

By default, the noise term is standard Brownian motion, i.e. additive Gaussian white noise
with identity covariance matrix. To construct different noise structures, see below.

## Noise term

The noise term can be specified via several keyword arguments. Based on these keyword
arguments, the noise function `g` is constructed behind the scenes unless explicitly given.

- The noise strength (i.e. the magnitude of the stochastic forcing) can be scaled with
  `noise_strength` (defaults to `1.0`). This factor is multiplied with the whole noise term.
- For non-diagonal and correlated noise, a covariance matrix can be provided via
  `covariance` (defaults to identity matrix of size `length(u0)`.)
- For more complicated noise structures, including state- and time-dependent noise, the
  noise function `g` can be provided explicitly as a keyword argument (defaults to
  `nothing`). For construction instructions, continue reading.

The function `g` interfaces to the diffusion function specified in an
[`SDEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/sde_types/#SciMLBase.SDEProblem)
of DynamicalSystems.jl. `g` must follow the same syntax as `f`, i.e., `g(u, p, t)`
for out-of-place (oop) and `g!(du, u, p, t)` for in-place (iip).

Unless `g` is of vector form and describes diagonal noise, a prototype type instance for the
output of `g` must be specified via the keyword argument `noise_prototype`. It can be of any
type `A` that has the method
[`LinearAlgebra.mul!(Y, A, B) -> Y`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.mul!)
defined.
Commonly, this is a matrix or sparse matrix. If this is not given, it
defaults to `nothing`, which means the `g` should be interpreted as being diagonal.

The noise process can be specified via `noise_process`. It defaults to a standard Wiener
process (Gaussian white noise).
For details on defining noise processes, see the docs of [DiffEqNoiseProcess.jl
](https://docs.sciml.ai/DiffEqNoiseProcess/stable/). A complete list of the pre-defined
processes can be found [here](https://docs.sciml.ai/DiffEqNoiseProcess/stable/noise_processes/).
Note that `DiffEqNoiseProcess.jl` also has an interface for defining custom noise processes.

By combining `g` and `noise_process`, you can define different types of stochastic systems.
Examples of different types of stochastic systems are listed on the
[StochasticDiffEq.jl tutorial page](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/sde_example/).
A quick overview of common types of stochastic systems can also be
found in the [online docs for `CoupledSDEs`](@ref defining-stochastic-dynamics).

## Keyword arguments

- `g`: noise function (default `nothing`)
- `noise_strength`: scaling factor for noise strength (default `1.0`)
- `covariance`: noise covariance matrix (default `nothing`)
- `noise_prototype`: prototype instance for the output of `g` (default `nothing`)
- `noise_process`: stochastic process as provided by [DiffEqNoiseProcess.jl](https://docs.sciml.ai/DiffEqNoiseProcess/stable/) (default `nothing`, i.e. standard Wiener process)
- `t0`: initial time (default `0.0`)
- `diffeq`: DiffEq solver settings (see below)
- `seed`: random number seed (default `rand(UInt64)`, so each `CoupledSDEs` produces a different noise realization unless a seed is given explicitly)

## DifferentialEquations.jl interfacing

The `CoupledSDEs` is evolved using the solvers of DifferentialEquations.jl.
To specify a solver via the `diffeq` keyword argument, use the flag `alg`, which can be
accessed after loading StochasticDiffEq.jl (`using StochasticDiffEq`).
The default `diffeq` is:

```julia
(alg = SOSRA(), abstol = 1.0e-2, reltol = 1.0e-2)
```

`diffeq` keywords can also include a `callback` for [event handling
](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

Dev note: `CoupledSDEs` is a light wrapper of  `SDEIntegrator` from StochasticDiffEq.jl.
The integrator is available as the field `integ`, and the `SDEProblem` is `integ.sol.prob`.
The convenience syntax `SDEProblem(ds::CoupledSDEs, tspan = (t0, Inf))` is available
to extract the problem.

## Converting between `CoupledSDEs` and `CoupledODEs`

You can convert a `CoupledSDEs` system to `CoupledODEs` to analyze its deterministic part
using the function `CoupledODEs(ds::CoupledSDEs; diffeq, t0)`.
Similarly, use `CoupledSDEs(ds::CoupledODEs [, p]; kwargs...)` to convert a `CoupledODEs` into
a `CoupledSDEs`.
"""
struct CoupledSDEs{IIP, D, I, P} <: ContinuousTimeDynamicalSystem
    # D parametrised by length of u0
    integ::I
    # things we can't recover from `integ`
    p0::P
    diffeq # isn't parameterized because it is only used for display
    noise_type::NamedTuple
end

###########################################################################################
# DiffEq options
###########################################################################################
# SOSRA is only applicable for additive  noise and must be adaptive
# default sciml tolerance is 1e-2
const DEFAULT_SDE_SOLVER = SOSRA()
const DEFAULT_STOCH_DIFFEQ_KWARGS = (abstol = 1.0e-2, reltol = 1.0e-2, dt = 0.1)
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

function CoupledSDEs(
        f,
        u0,
        p = SciMLBase.NullParameters();
        g = nothing,
        noise_strength = 1.0,
        covariance = nothing,
        t0 = 0.0,
        diffeq = DEFAULT_STOCH_DIFFEQ,
        noise_prototype = nothing,
        noise_process = nothing,
        seed = rand(UInt64)
    )
    IIP = isinplace(f, 4) # from SciMLBase
    if !isnothing(g)
        @assert IIP == isinplace(g, 4) "f and g must both be in-place or out-of-place"
    end

    noise_type, cov = find_noise_type(g, u0, p, t0, noise_process, covariance, noise_prototype, IIP)
    g, noise_prototype = construct_diffusion_function(
        g, cov, noise_prototype, noise_strength, length(u0), IIP
    )

    s = correct_state(Val{IIP}(), u0)
    T = eltype(s)
    prob = SDEProblem{IIP}(
        f,
        g,
        s,
        (T(t0), T(1.0e11)),
        p;
        noise_rate_prototype = noise_prototype,
        noise = noise_process,
        seed = seed
    )
    return CoupledSDEs(prob, diffeq, noise_type)
end

function CoupledSDEs(
        prob::SDEProblem, diffeq = DEFAULT_STOCH_DIFFEQ, noise_type = nothing; special_kwargs...
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
        prob = SciMLBase.remake(prob; tspan = (U(0), U(Inf)))
    end
    if isnothing(noise_type)
        noise_type, _ = find_noise_type(prob, IIP)
    end

    solver, remaining = _decompose_into_sde_solver_and_remaining(diffeq)
    # The default `dtmin` from SciML scales with `tspan`. With our open-ended
    # `tspan = (0, 1e11)` it becomes ~1e-5, which is too coarse for the SDE
    # adaptive controller and causes spurious `DtLessThanMin` aborts.
    remaining = haskey(remaining, :dtmin) ? remaining : merge((dtmin = 0.0,), remaining)
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
    return CoupledSDEs{IIP, D, typeof(integ), P}(
        integ, deepcopy(prob.p), diffeq, noise_type
    )
end
# This preserves the referenced MTK system and the originally passed diffeq kwargs
CoupledSDEs(ds::CoupledSDEs, diffeq) = CoupledSDEs(SDEProblem(ds), merge(ds.diffeq, diffeq))

"""
    reinit!(ds::CoupledSDEs, u = initial_state(ds); kwargs...) → ds

Re-initialize a [`CoupledSDEs`](@ref). In addition to the keywords accepted by
`reinit!` for any [`DynamicalSystem`](@ref), the following is supported:

- `seed::UInt64`: re-seed the noise process random number generator. Defaults to
  `rand(UInt64)`, so by default every `reinit!` produces a fresh, independent noise
  realization. Pass an explicit `seed` to obtain a reproducible noise stream.
"""
function SciMLBase.reinit!(
        ds::CoupledSDEs, u::AbstractArray = initial_state(ds);
        p = current_parameters(ds),
        t0 = initial_time(ds),
        seed::UInt64 = rand(UInt64), kw...
    )
    set_parameters!(ds, p)
    # Adaptive solvers carry the previous run's last `dt`; resetting it ensures
    # `seed` reproducibility. Non-adaptive solvers must keep their user-specified `dt`.
    reset_dt = ds.integ.opts.adaptive
    SciMLBase.reinit!(ds.integ, u; reinit_dae = false, reset_dt, t0, kw...)
    Random.seed!(ds.integ.W.rng, seed)
    setup_next_step!(ds.integ.W, ds.integ.u, ds.integ.p)
    return ds
end

"""
    CoupledSDEs(ds::CoupledODEs, p = current_parameters(ds); kwargs...)

Converts a [`CoupledODEs`](@ref) into a [`CoupledSDEs`](@ref).
While `p` is optional, it is likely to change as the
diffusion (noise) function `g` is added.
"""
function CoupledSDEs(
        ds::CoupledODEs,
        p = current_parameters(ds); # the parameter is likely changed as the diffusion function g is added.
        g = nothing,
        noise_strength = 1.0,
        covariance = nothing,
        diffeq = DEFAULT_STOCH_DIFFEQ,
        noise_prototype = nothing,
        noise_process = nothing,
        seed = rand(UInt64)
    )
    prob = referenced_sciml_prob(ds)
    # we want the symbolic jacobian to be transfered over
    # dynamic_rule(ds) takes the deepest nested f wich does not have a jac field
    return CoupledSDEs(
        prob.f, current_state(ds), p;
        g, noise_strength, covariance, diffeq, noise_prototype, noise_process, seed
    )
end

"""
    CoupledODEs(ds::CoupledSDEs; kwargs...)

Converts a [`CoupledSDEs`](@ref) into a [`CoupledODEs`](@ref) by extracting the
deterministic part of `ds`.
"""
function CoupledODEs(
        sys::CoupledSDEs; diffeq = DEFAULT_DIFFEQ, t0 = 0.0
    )
    prob = referenced_sciml_prob(sys)
    # we want the symbolic jacobian to be transfered over
    # dynamic_rule(ds) takes the deepest nested f wich does not have a jac field
    return CoupledODEs(
        prob.f, SVector{length(sys.integ.u)}(sys.integ.u), sys.p0; diffeq = diffeq, t0 = t0
    )
end

# Pretty print
function additional_details(ds::CoupledSDEs)
    solver, remaining = _decompose_into_sde_solver_and_remaining(ds.diffeq)
    return [
        "SDE solver" => string(nameof(typeof(solver))),
        "SDE kwargs" => remaining,
        "Noise type" => ds.noise_type,
    ]
end

###########################################################################################
# API - obtaining information from the system
###########################################################################################

SciMLBase.isinplace(::CoupledSDEs{IIP}) where {IIP} = IIP
StateSpaceSets.dimension(::CoupledSDEs{IIP, D}) where {IIP, D} = D
current_state(ds::CoupledSDEs) = current_state(ds.integ)
isdeterministic(ds::CoupledSDEs) = false

set_state!(ds::CoupledSDEs, u::AbstractArray) = (set_state!(ds.integ, u); ds)
SciMLBase.step!(ds::CoupledSDEs, args...) = (step!(ds.integ, args...); ds)

# TODO We have to check if for SDEIntegrators a different step ReturnCode is possible.
function successful_step(integ::AbstractSDEIntegrator)
    rcode = integ.sol.retcode
    return rcode == SciMLBase.ReturnCode.Default || rcode == SciMLBase.ReturnCode.Success
end

# This is here to ensure that `derivative_discontinuity!` is called
function set_parameter!(ds::CoupledSDEs, args...)
    _set_parameter!(ds, args...)
    derivative_discontinuity!(ds.integ, true)
    return nothing
end

referenced_sciml_prob(ds::CoupledSDEs) = ds.integ.sol.prob

"""
    diffusion_matrix(ds::CoupledSDEs)

Returns the diffusion matrix of the stochastic term of the [`CoupledSDEs`](@ref) `ds`,
provided that the diffusion function `g` can be expressed as a constant invertible matrix.
If this is not the case, returns `nothing`.

Note: The diffusion matrix ``Σ`` is the square root of the noise covariance matrix ``Q`` (see
[`covariance_matrix`](@ref)), defined via the Cholesky decomposition ``Q = Σ Σ^\\top``.
"""
function diffusion_matrix(ds::CoupledSDEs{IIP, D})::AbstractMatrix where {IIP, D}
    if ds.noise_type[:invertible]
        diffusion = diffusion_function(ds)
        A = diffusion(zeros(D), current_parameters(ds), 0.0)
        A = A isa AbstractMatrix ? A : Diagonal(A)
    else
        @warn """
        The diffusion function of the `CoupledSDEs` cannot be expressed as a constant
        invertible matrix.
        """
        A = nothing
    end
    return A
end

"""
    covariance_matrix(ds::CoupledSDEs)

Returns the covariance matrix of the stochastic term of the [`CoupledSDEs`](@ref) `ds`,
provided that the diffusion function `g` can be expressed as a constant invertible matrix.
If this is not the case, returns `nothing`.

See also [`diffusion_matrix`](@ref).
"""
function covariance_matrix(ds::CoupledSDEs)::AbstractMatrix
    A = diffusion_matrix(ds)
    return (A == nothing) ? nothing : A * A'
end

###########################################################################################
# Utilities
###########################################################################################

"""
    construct_diffusion_function(g, covariance, noise_prototype, noise_strength, D, IIP)

Constructs the noise function `g` based on the keyword arguments
`g`, `covariance`, `noise_prototype`, and `noise_strength`
specified by the user when defining a `CoupledSDEs`.

Here `D` is the system dimension and `IIP` indicated whether the function `g` is in-place
(`IIP = true`) or out-of-place (`false`).

Returns `g, noise_prototype`.
"""
function construct_diffusion_function(
        g, covariance, noise_prototype, noise_strength, D, IIP
    )
    if isnothing(g) # diagonal additive noise
        cov = isnothing(covariance) ? LinearAlgebra.I(D) : covariance
        size(cov) != (D, D) &&
            throw(ArgumentError("Covariance matrix must be of size $((D, D))"))
        A = sqrt(cov)
        if IIP
            if isdiag(cov)
                diag_const = collect(noise_strength .* diag(A))
                g = let diag_const = diag_const
                    (du, u, p, t) -> (du .= diag_const; nothing)
                end
            else
                A_const = collect(noise_strength .* A)
                g = let A_const = A_const
                    (du, u, p, t) -> (du .= A_const; nothing)
                end
                noise_prototype = zeros(size(A))
                # ^ we could make this sparse to make it more performant
            end
        else
            if isdiag(cov)
                diag_const = SVector{D, eltype(A)}(diag(noise_strength .* A))
                g = let diag_const = diag_const
                    (u, p, t) -> diag_const
                end
            else
                A_const = SMatrix{size(A)..., eltype(A)}(noise_strength .* A)
                g = let A_const = A_const
                    (u, p, t) -> A_const
                end
                noise_prototype = zeros(size(A))
            end
        end
    end
    return g, noise_prototype
end


"""
Checks whether the function g depends on u based on 10 random points around the given u.
"""
function is_state_independent(g, u, p, t)
    rdm_states = [u .+ rand(eltype(u), length(u)) .- 0.5 for _ in 1:10]
    val = map(u -> g(u, p, t), rdm_states)
    return length(unique(val)) == 1
end

"""
Checks whether g depends explicitly on time for select points on the interval [t0, t0+101].
"""
function is_time_independent(g, u, p, t0)
    trange = t0 .+ [0.0, 0.101, 1.01, 10.1, 101.0]
    val = map(t -> g(u, p, t), trange)
    return length(unique(val)) == 1
end

"""
Checks whether a matrix x is invertible by verifying that it has nonzero determinant.
"""
function is_invertible(x; tol = 1.0e-10)
    F = lu(x, check = false)
    det = abs(prod(diag(F.U)))
    return det > tol
end

"""
Checks whether the function f is linear by checking additivity and scalar multiplication
for two points x and y.
"""
function is_linear(f, x, y, c)
    check1 = f(x + y) == f(x) + f(y)
    check2 = f(c * x) == c * f(x)
    return check1 && check2
end

function diffusion_function(g, IIP, noise_prototype)
    return function diffusion(u, p, t)
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
    prob = referenced_sciml_prob(ds)
    return diffusion_function(prob.g, IIP, prob.noise_rate_prototype)
end

"""
Classifies the noise type of the CoupledSDEs given by the user.
Returns a named tuple of noise properties and the noise covariance matrix (if applicable).
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

        islinear = true
        if !state_independent
            for i in 1:10
                check = is_linear(
                    u -> diffusion(u, p, t0),
                    u0 + i .* rand(D), u0 + i .* rand(D), 2.0
                )
                check ? nothing : islinear = false
            end
        end

        # Previous formulation:
        #islinear = !state_independent ?
        #           is_linear(u -> diffusion(u, p, t0), rand(D), rand(D), 2.0) : true

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

    noise_type = (
        additive = isadditive, autonomous = isautonomous,
        linear = islinear, invertible = isinvertible,
    )
    return noise_type, covariance
end

function find_noise_type(prob::SDEProblem, IIP)
    return find_noise_type(
        prob.g, prob.u0, prob.p, prob.tspan[1], prob.noise,
        nothing, prob.noise_rate_prototype, IIP
    )
end
