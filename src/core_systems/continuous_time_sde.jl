export CoupledSDEs

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
(alg = SOSRI(), abstol = 1.0e-6, reltol = 1.0e-6)
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
    noise_type::NamedTuple
end
