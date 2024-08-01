# `CoupledSDEs`

```@docs
CoupledSDEs
```


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

## [Examples defining stochastic dynamics](@id defining-stochastic-dynamics)

TODO?
