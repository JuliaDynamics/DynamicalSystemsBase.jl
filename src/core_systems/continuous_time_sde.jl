export CoupledSDEs

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
- `seed`: random number seed (default `UInt64(0)`)

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
Similarly, use `CoupledSDEs(ds::CoupledODEs, p; kwargs...)` to convert a `CoupledODEs` into
a `CoupledSDEs`.
"""
struct CoupledSDEs{IIP,D,I,P} <: ContinuousTimeDynamicalSystem
    # D parametrised by length of u0
    integ::I
    # things we can't recover from `integ`
    p0::P
    diffeq # isn't parameterized because it is only used for display
    noise_type::NamedTuple
end

function SciMLBase.reinit!(ds::CoupledSDEs, u::AbstractArray = initial_state(ds);
    p = current_parameters(ds), t0 = initial_time(ds), kw...
  )
  set_parameters!(ds, p)
  reinit!(ds.integ, u; t0, kw...)
  return ds
end