# Define a CoupledSDEs system

A `CoupledSDEs` defines a stochastic dynamical system of the form

```math
\text{d}\vec x = f(\vec x(t); \ p)  \text{d}t + \sigma g(\vec x(t);  \ p) \, \text{d}\mathcal{W} \ ,
```
where $\vec x \in \mathbb{R}^\text{D}$, $\sigma > 0$ is the noise strength, $\text{d}\mathcal{W}=\Gamma \cdot \text{d}\mathcal{N}$, and $\mathcal N$ denotes a stochastic process. The (positive definite) noise covariance matrix is $\Sigma = \Gamma \Gamma^\top \in \mathbb R^{N\times N}$.

The function $f$ is the deterministic part of the system and follows the syntax of a `ContinuousTimeDynamicalSystem` in [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/tutorial/), i.e., `f(u, p, t)` for out-of-place (oop) and `f!(du, u, p, t)` for in-place (iip). The function $g$ allows to specify the stochastic dynamics of the system along with the [noise process](#noise-process) $\mathcal{W}$. It should be of the same type (iip or oop) as $f$.

By combining $\sigma$, $g$ and $\mathcal{W}$, you can define different type of stochastic systems. Examples of different types of stochastic systems can be found on the [StochasticDiffEq.jl tutorial page](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/sde_example/). A quick overview of common types of stochastic systems can be found [below](#Defining-stochastic-dynamics).

!!! info
    Note that nonlinear mixings of the Noise Process $\mathcal{W}$ fall into the class of random ordinary differential equations (RODEs) and is not supported in DynamicalSystems.jl at the moment. See [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/rode_example/) for an implementation.

```@docs
CoupledSDEs
```

## Converting between `CoupledSDEs` and `CoupledODEs`

!!! tip "Analyzing deterministic dynamics with DynamicalSystems.jl"
    The deterministic part of a [`CoupledSDEs`](@ref) system can easily be extracted as a 
    [`CoupledODEs`](https://juliadynamics.github.io/DynamicalSystems.jl/dev/tutorial/#DynamicalSystemsBase.CoupledODEs), a common subtype of a `ContinuousTimeDynamicalSystem` in DynamicalSystems.jl.

- `CoupledODEs(sde::CoupledSDEs)` extracts the deterministic part of `sde` as a `CoupledODEs`
- `CoupledSDEs(ode::CoupledODEs, g)`, with `g` the noise function, turns `ode` into a `CoupledSDEs`

```@docs
CoupledODEs(::CoupledSDEs; kwargs...)
```
For example, the
Lyapunov spectrum of a `CoupledSDEs` in the absence of noise, here exemplified by the
FitzHugh-Nagumo model, can be computed by typing:

```julia
using DynamicalSystemsBase, StaticArrays
using DynamicalSystemsBase: idfunc

function fitzhugh_nagumo(u, p, t)
    x, y = u
    ϵ, β, α, γ, κ, I = p

    dx = (-α * x^3 + γ * x - κ * y + I) / ϵ
    dy = -β * y + x

    return SA[dx, dy]
end

sde = CoupledSDEs(fitzhugh_nagumo, idfunc, zeros(2), [1.,3.,1.,1.,1.,0.], 0.1)
tr = trajectory(sde, 10)
```

## Available noise processes
We provide the noise processes $\mathcal{W}$ that can be used in the stochastic simulations through the [DiffEqNoiseProcess.jl](https://docs.sciml.ai/DiffEqNoiseProcess/stable) package. A complete list of the available processes can be found [here](https://docs.sciml.ai/DiffEqNoiseProcess/stable/noise_processes/).
```


