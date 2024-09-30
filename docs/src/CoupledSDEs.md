# `CoupledSDEs`

```@docs
CoupledSDEs
```

## [Examples defining stochastic dynamics](@id defining-stochastic-dynamics)

Let's look at some examples of the different types of stochastic systems that can be defined.

For simplicity, we choose a slow exponential growth in 2 dimensions as the deterministic dynamics `f`:
```@example type
using DynamicalSystemsBase, StochasticDiffEq, DiffEqNoiseProcess
using CairoMakie
import Random # hide
Random.seed!(10) # hide
f!(du, u, p, t) = du .= 1.01u # deterministic part

function plot_trajectory(Y, t)
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel = "time", ylabel = "variable")
    for var in columns(Y)
        lines!(ax, t, var)
    end
    fig
end;
```

### Additive noise
When $g(u, p, t)$ is independent of the state $u$, the noise is called additive; otherwise, it is multiplicative.
We can define a simple additive noise system as follows:
```@example type
sde = CoupledSDEs(f!, zeros(2));
```
which is equivalent to
```@example type
t0 = 0.0; W0 = zeros(2);
W = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, zeros(2);
    noise_process=W, covariance=[1 0; 0 1], noise_strength=1.0
    );
```
We defined a Wiener process `W`, whose increments are vectors of normally distributed random numbers of length matching the output of `g`. The noise is applied element-wise, i.e., `g.*dW`. Since the noise processes are uncorrelated, meaning the covariance matrix is diagonal, this type of noise is referred to as diagonal.

We can sample a trajectory from this system using the `trajectory` function also used for the deterministic systems:
```@example type
tr = trajectory(sde, 1.0)
plot_trajectory(tr...)
```

#### Correlated noise
In the case of correlated noise, the random numbers in a vector increment `dW` are correlated. This can be achieved by specifying the covariance matrix $\Sigma$ via the `covariance` keyword:
```@example type
ρ = 0.3
Σ = [1 ρ; ρ 1]
diffeq = (alg = LambaEM(), dt=0.1)
sde = CoupledSDEs(f!, zeros(2); covariance=Σ, diffeq=diffeq)
```
Alternatively, we can parametrise the covariance matrix by defining the diffusion function $g$ ourselves:
```@example type
g!(du, u, p, t) = (du .= [1 p[1]; p[1] 1]; return nothing) 
sde = CoupledSDEs(f!, zeros(2), (ρ); g=g!, noise_prototype=zeros(2, 2))
```
Here, we had to provide `noise_prototype` to indicate that the diffusion function `g` will output a 2x2 matrix.

#### Scalar noise
If all state variables are forced by the same single random variable, we have scalar noise.
To define scalar noise, one has to give an one-dimensional noise process to the `noise_process` keyword of the `CoupledSDEs` constructor. 
```@example type
t0 = 0.0; W0 = 0.0;
noise = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, rand(2)/10; noise_process=noise)

tr = trajectory(sde, 1.0)
plot_trajectory(tr...)
```
We can see that noise applied to each variable is the same.

### Multiplicative and time-dependent noise
In the SciML ecosystem, multiplicative noise is defined through the condition $g_i(t, u)=a_i u$. However, in the literature the name is more broadly used for any situation where the noise is non-additive and depends on the state $u$, possibly also in a non-linear way. When defining a `CoupledSDEs`, we can make the noise term time- and state-dependent by specifying an explicit time- or state-dependence in the noise function `g`, just like we would define `f`. For example, we can define a system with temporally decreasing multiplicative noise as follows:
```@example type
function g!(du, u, p, t)
    du .= u ./ (1+t)
    return nothing
end
sde = CoupledSDEs(f!, rand(2)./10; g=g!)
```

#### Non-diagonal noise
Non-diagonal noise allows for the terms to be linearly mixed (correlated) via `g` being a matrix. Suppose we have two Wiener processes and two state variables such that the output of `g` is a 2x2 matrix. Therefore, we have
```math
du_1 = f_1(u,p,t)dt + g_{11}(u,p,t)dW_1 + g_{12}(u,p,t)dW_2 \\
du_2 = f_2(u,p,t)dt + g_{21}(u,p,t)dW_1 + g_{22}(u,p,t)dW_2
```
To indicate the structure that `g` should have, we must use the `noise_prototype` keyword. Let us define a special type of non-diagonal noise called commutative noise. For this we can utilize the `RKMilCommute` algorithm which is designed to utilize the structure of commutative noise.

```@example type
σ = 0.25 # noise strength
function g!(du, u, p, t)
  du[1,1] = σ*u[1]
  du[2,1] = σ*u[2]
  du[1,2] = σ*u[1]
  du[2,2] = σ*u[2]
    return nothing
end
diffeq = (alg = RKMilCommute(), reltol = 1e-3, abstol = 1e-3, dt=0.1)
sde = CoupledSDEs(f!, rand(2)./10; g=g!, noise_prototype = zeros(2, 2), diffeq = diffeq)
```

!!! warning
    Non-diagonal problems need specific solvers. See the [SciML recommendations](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/#sde_solve).