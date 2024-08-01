export CoupledSDEs

"""
    CoupledSDEs <: ContinuousTimeDynamicalSystem
    CoupledSDEs(f, g, u0 [, p, σ]; kwargs...)

A stochastic continuous time dynamical system defined by a set of
coupled stochastic differential equations as follows:
```math
d\\vec{u} = \\vec{f}(\\vec{u}, p, t) dt + σ \\vec{g}(\\vec{u}, p, t) dW_t
```
where
``\\sigma > 0`` is the overall noise strength,
``\\text{d}\\mathcal{W} = \\Gamma \\cdot \\text{d}\\mathcal{N}``,
and ``\\mathcal N`` denotes a stochastic process.
The (positive definite) noise covariance matrix is
``\\Sigma = \\Gamma \\Gamma^\\top \\in \\mathbb R^{N\\times N}``.

Optionally you can provide a paraemter container and the overall noise strength `σ`.
and initial time as keyword `t0`. By default `p = nothing`, `σ = 1` and `t0 = 0`.

For construction instructions regarding `f, u0` see the [DynamicalSystems.jl tutorial
](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/dynamicalsystems/dev/tutorial/).
`g` must follow the sae syntax, i.e., `g(u, p, t)` for out-of-place (oop) and
`g(du, u, p, t)` for in-place (iip).

## Stochastic part

The stochastic part of the differential equation is defined by the function `g` and
the keyword arguments `noise_rate_prototype` and `noise`. `noise` indicates
the noise process applied and defaults to Gaussian white noise (Wiener process).
For details on defining various noise processes, refer to [DiffEqNoiseProcess.jl
](https://docs.sciml.ai/DiffEqNoiseProcess/stable/).

`noise_rate_prototype` indicates the prototype type instance for the noise rates, i.e.,
the output of `g`. It can be any type which overloads `A_mul_B!` with itself being the
middle argument. Commonly, this is a matrix or sparse matrix. If this is not given, it
defaults to `nothing`, which means the `g` should be interpreted as being diagonal.

By combining ``\\sigma g, \\mathcal{W}``, you can define different type of stochastic
systems. Examples of different types of stochastic systems can be found on the
[StochasticDiffEq.jl tutorial page](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/sde_example/).
A quick overview of common types of stochastic systems can also be
found in the [online docs for `CoupledSDEs`](@ref defining-stochastic-dynamics).

The noise processes ``\\mathcal{W}`` that can be used in the stochastic simulations are
obtained from [DiffEqNoiseProcess.jl](https://docs.sciml.ai/DiffEqNoiseProcess/stable).
A complete list of the available processes can be found [here](https://docs.sciml.ai/DiffEqNoiseProcess/stable/noise_processes/).

## DifferentialEquations.jl interfacing

The ODEs are evolved via the solvers of DifferentialEquations.jl.
If you want to specify a solver, do so by using the keyword `alg`,
which requires you to be using StochasticDiffEq.jl to access the solvers.
The default `diffeq` is:

```julia
(alg = SOSRI(), abstol = 1.0e-6, reltol = 1.0e-6)
```

`diffeq` keywords can also include `callback` for [event handling
](https://docs.sciml.ai/DiffEqDocs/stable/features/callback_functions/).

Dev note: `CoupledSDEs` is a light wrapper of  `SDEIntegrator` from StochasticDiffEq.jl.
The integrator is available as the field `integ`, and the `SDEProblem` is `integ.sol.prob`.
The convenience syntax `SDEProblem(ds::CoupledSDEs, tspan = (t0, Inf))` is available
to extract the problem.

## Converting between `CoupledSDEs` and `CoupledODEs`

You can convert to `CoupledODEs` to analyze the deterministic part via
the function `CoupledODEs(ds::CoupledSDEs; diffeq, t0)`.
Similarly, use `CoupledSDEs(ds::CoupledODEs, g; kw...)` to convert to a stochastic system.


"""
struct CoupledSDEs{IIP,D,I,P,S} <: ContinuousTimeDynamicalSystem
    # D parametrised by length of u0
    # S indicated if the noise strength has been added to the diffusion function
    integ::I
    # things we can't recover from `integ`
    p0::P
    noise_strength
    diffeq # isn't parameterized because it is only used for display
    noise_type::NamedTuple
end
