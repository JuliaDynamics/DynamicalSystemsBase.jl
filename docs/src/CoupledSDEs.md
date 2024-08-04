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
idfunc!(du, u, p, t) = (du .= ones(eltype(u), length(u)); return nothing)
σ = 0.25 # noise strength

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
When $g$ is independent of the state variables $u$, the noise is called additive, otherwise it is called multiplicative. The ladder is somewhat misleading, as it has come to signify the general case, despite seeming to suggest the limited case where $g(x) \propto x$. We can define a simple additive noise system as follows:
```@example type
t0 = 0.0; W0 = zeros(2);
W = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, idfunc!, zeros(2); noise=W)
```
which is equivalent to
```@example type
sde = CoupledSDEs(f!, idfunc!, zeros(2));
```
We defined a vector of random numbers `dW`, whose size matches the output of `g`, with the noise applied element-wise, i.e., `g.*dW`. Specifically, `dW` is a vector of Gaussian noise processes whose size matches the output of `g`. The noise is applied element-wise, i.e., `g.*dW`. By default, the noise processes are uncorrelated, meaning the covariance matrix is diagonal. This type of noise is referred to as diagonal. The vector `dW` is by default zero mean white gaussian noise $\mathcal{N}(0, \text{d}t)$ where the variance is the timestep $\text{d}t$ unit variance (Wiener Process). We can sample a trajectory from this system using the `trajectory` also used for the deterministic systems:
```@example type
tr = trajectory(sde, 1.0)
plot_trajectory(tr...)
```

#### Correlated noise
Correlated noise is where the random number generated in `dW` are correlated. This can be done by specifying the covariance matrix $\Sigma$ for the noise processes and passing it to a `CorrelatedWienerProcess` of `DiffEqNoiseProcess.jl`.:
```@example type
ρ = 0.3
Σ = [1 ρ; ρ 1]
t0 = 0.0; W0 = zeros(2); Z0 = zeros(2);
W = CorrelatedWienerProcess(Σ, t0, W0, Z0)
```
Indeed, sampling 1_000 values from this process and plotting them shows the correlation:
```@example type
prob = NoiseProblem(W, (0.0, 1.0))
sol = solve(prob; dt = 0.1)

output_func = (sol, i) -> (sol.dW, false)
ensemble_prob = EnsembleProblem(prob, output_func = output_func)
values = Array(solve(ensemble_prob, dt = 0.1, trajectories = 10_000))
scatter(values[1,:], values[2,:])
```
The CoupledSDEs can be defined by passing the `CorrelatedWienerProcess` to the `noise` keyword:
```@example type
sde = CoupledSDEs(f!, idfunc!, zeros(2); noise=W)
```
The covariance matrix $\Sigma$ can be accessed via `sde.integ.covariance`.

!!! warning
    Another way to define correlated noise is to use the diffusion function $g$ to define the matrix $\Sigma$. However, this is in general not recommended as the solvers from StochasticDiffEq will assume that the problem is more complicated such that the fast additive solvers like `SOSRA` and `SOSRI` will not work.

#### Scalar noise
Scalar noise is where a single random variable is applied to all dependent variables. To do this, one has to give an one dimensional noise process to the `noise` keyword of the `CoupledSDEs` constructor. 
```@example type
t0 = 0.0; W0 = 0.0;
noise = WienerProcess(t0, W0, 0.0)
sde = CoupledSDEs(f!, idfunc!, rand(2)/10; noise=noise)

tr = trajectory(sde, 1.0)
plot_trajectory(tr...)
```
We can see that noise applied to each variable is the same.


### Multiplicative and non-autonomous noise
Multiplicative noise in the SciML ecosystem is defined as when $g_i(t, u)=a_i u$. However, in the literature the name is sometimes used for when the noise is not-additive, including nonlinear. In general, we can make the noise time- and state-dependent in a similar fashion to the deterministic dynamics. For example, we can define a system with decreasing multiplicative noise as follows:
```@example type
function g!(du, u, p, t)
    du .= u ./ (1+t)
    return nothing
end
sde = CoupledSDEs(f!, g!, rand(2)./10)
```

#### Non-diagonal noise
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
diffeq = (alg = RKMilCommute(), reltol = 1e-3, abstol = 1e-3)
sde = CoupledSDEs(f!, g, rand(2)./10; noise_rate_prototype = zeros(2, 2), diffeq = diffeq)
```

!!! warning
    Non-diagonal problems need specific type of solvers. See the [SciML recommendations](https://docs.sciml.ai/DiffEqDocs/stable/solvers/sde_solve/#sde_solve).
