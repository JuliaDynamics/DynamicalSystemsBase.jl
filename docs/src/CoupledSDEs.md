# Define a CoupledSDEs system

A `CoupledSDEs` defines a stochastic dynamical system of the form

```math
\text{d}\vec x = f(\vec x(t); \ p)  \text{d}t + \sigma g(\vec x(t);  \ p) \, \text{d}\mathcal{W} \ ,
```
where $\vec x \in \mathbb{R}^\text{D}$, $\sigma > 0$ is the noise strength, $\text{d}\mathcal{W}=\Gamma \cdot \text{d}\mathcal{N}$, and $\mathcal N$ denotes a stochastic process. The (positive definite) noise covariance matrix is $\Sigma = \Gamma \Gamma^\top \in \mathbb R^{N\times N}$.

The function $f$ is the deterministic part of the system and follows the syntax of a `ContinuousTimeDynamicalSystem` in [DynamicalSystems.jl](https://juliadynamics.github.io/DynamicalSystems.jl/latest/tutorial/), i.e., `f(u, p, t)` for out-of-place (oop) and `f!(du, u, p, t)` for in-place (iip). The function $g$ allows to specify the stochastic dynamics of the system along with the [noise process](#noise-process) $\mathcal{W}$. It should be of the same type (iip or oop) as $f$.

By combining $\sigma$, $g$ and $\mathcal{W}$, you can define different type of stochastic systems. Examples of different types of stochastic systems can be found on the [StochasticDiffEq.jl tutorial page](https://docs.sciml.ai/DiffEqDocs/stable/tutorials/sde_example/). A quick overview of common types of stochastic systems can be found [below](#Type-of-stochastic-system).

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
CoupledODEs
```
For example, the
Lyapunov spectrum of a `CoupledSDEs` in the absence of noise, here exemplified by the
FitzHugh-Nagumo model, can be computed by typing:

```julia
using CriticalTransitions
using DynamicalSystems: lyapunovspectrum

function fitzhugh_nagumo(u, p, t)
    x, y = u
    ϵ, β, α, γ, κ, I = p

    dx = (-α * x^3 + γ * x - κ * y + I) / ϵ
    dy = -β * y + x

    return SA[dx, dy]
end

sys = CoupledSDEs(fitzhugh_nagumo, idfunc, zeros(2), [1.,3.,1.,1.,1.,0.], 0.1)
ls = lyapunovspectrum(CoupledODEs(sys), 10000)
```

## Defining stochastic dynamics
Let's look at some examples of the different types of stochastic systems that can be defined.

For simplicity, we choose a slow exponential growth in 2 dimensions as the deterministic dynamics `f`:
```@example type
using CriticalTransitions, Plots
import Random # hide
Random.seed!(10) # hide
f!(du, u, p, t) = du .= 1.01u # deterministic part
σ = 0.25 # noise strength
```
### Additive noise
When `g \, \text{d}\mathcal{W}` is independent of the state variables `u`, the noise is called additive.

#### Diagonal noise
A system of diagonal noise is the most common type of noise. It is defined by a vector of random numbers `dW` whose size matches the output of `g` where the noise is applied element-wise, i.e. `g.*dW`.
```@example type
t0 = 0.0; W0 = zeros(2);
W = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise=W)
```
or equivalently
```@example type
sde = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ)
```
where we used the helper function
```@docs
idfunc!
idfunc
```
The vector `dW` is by default zero mean white gaussian noise $\mathcal{N}(0, \text{d}t)$ where the variance is the timestep $\text{d}t$ unit variance (Wiener Process).
```@example type
sol = simulate(sde, 1.0, dt=0.01, alg=SOSRA())
plot(sol)
```

#### Scalar noise
Scalar noise is where a single random variable is applied to all dependent variables. To do this, one has to give the noise process to the `noise` keyword of the `CoupledSDEs` constructor. A common example is the Wiener process starting at `W0=0.0` at time `t0=0.0`.

```@example type
t0 = 0.0; W0 = 0.0;
noise = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, idfunc!, rand(2)./10, nothing, σ; noise=noise)
sol = simulate(sde, 1.0, dt=0.01, alg=SOSRA())
plot(sol)
```
### Multiplicative noise
Multiplicative Noise is when $g_i(t, u)=a_i u$
```@example type
function g(du, u, p, t)
    du[1] = σ*u[1]
    du[2] = σ*u[2]
    return nothing
end
sde = CoupledSDEs(f!, g, rand(2)./10)
sol = simulate(sde, 1.0, dt=0.01, alg=SOSRI())
plot(sol)
```
### Correlated noise
```@example type
ρ = 0.3
Σ = [1 ρ; ρ 1]
t0 = 0.0; W0 = zeros(2); Z0 = zeros(2);
W = CorrelatedWienerProcess(Σ, t0, W0, Z0)
sde = CoupledSDEs(f!, idfunc!, zeros(2), nothing, σ; noise=W)
sol = simulate(sde, 1.0, dt=0.01, alg=SOSRA())
plot(sol)
```

#### More complex Non-diagonal noise
Non-diagonal noise allows for the terms to linearly mixed via `g` being a matrix. Suppose we have two Wiener processes and two dependent random variables such that the output of `g` is a 2x2 matrix. Therefore, we have
```math
du_1 = f_1(u,p,t)dt + g_{11}(u,p,t)dW_1 + g_{12}(u,p,t)dW_2 \\
du_2 = f_2(u,p,t)dt + g_{21}(u,p,t)dW_1 + g_{22}(u,p,t)dW_2
```
To indicate the structure that `g` should have, we can use the `noise_rate_prototype` keyword. Let us define a special type of non-diagonal noise called commutative noise. For this we can utilize the `RKMilCommute` algorithm which is designed to utilise the structure of commutative noise.

```@example type
function g(du, u, p, t)
  du[1,1] = σ*u[1]
  du[2,1] = σ*u[2]
  du[1,2] = σ*u[1]
  du[2,2] = σ*u[2]
    return nothing
end
sde = CoupledSDEs(f!, g, rand(2)./10, noise_rate_prototype = zeros(2, 2))
sol = simulate(sde, 1.0, dt=0.01, alg=RKMilCommute())
plot(sol)
```

!!! warning
    Non-diagonal problems need specific type of solvers. See the [SciML recommendations](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/#sde_solve).

## Available noise processes
We provide the noise processes $\mathcal{W}$ that can be used in the stochastic simulations through the [DiffEqNoiseProcess.jl](https://docs.sciml.ai/DiffEqNoiseProcess/stable) package. A complete list of the available processes can be found [here](https://docs.sciml.ai/DiffEqNoiseProcess/stable/noise_processes/).
```


