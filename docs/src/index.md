# DynamicalSystemsBase.jl

```@docs
DynamicalSystemsBase
```

## The `DynamicalSystem` API

```@docs
DynamicalSystem
```

```@docs
current_state
initial_state
current_parameters
initial_parameters
isdeterministic
isdiscretetime
dynamic_rule
current_time
initial_time
isinplace(::DynamicalSystem)
```

```@docs
reinit!(::DynamicalSystem, args...; kwargs...)
set_state!
set_parameter!
set_parameters!
```

## Time evolution
```@docs
step!(::DynamicalSystem, args...; kwargs...)
trajectory
StateSpaceSet
```

## `DeterministicIteratedMap`
```@docs
DeterministicIteratedMap
```

## `CoupledODEs`
```@docs
CoupledODEs
```

## `StroboscopicMap`
```@docs
StroboscopicMap
```

## `PoincareMap`
```@docs
PoincareMap
current_crossing_time
poincaresos
```

## `ProjectedDynamicalSystem`
```@docs
ProjectedDynamicalSystem
```

## `ParallelDynamicalSystem`
```@docs
ParallelDynamicalSystem
initial_states
current_states
```

## `TangentDynamicalSystem`
```@docs
CoreDynamicalSystem
TangentDynamicalSystem
current_deviations
set_deviations!
orthonormal
```

## Parallelization

Since `DynamicalSystem`s are mutable, one needs to copy them before parallelizing, to avoid having to deal with complicated race conditions etc. The simplest way is with `deepcopy`. Here is an example block that shows how to parallelize calling some expensive function (e.g., calculating the Lyapunov exponent) over a parameter range using `Threads`:

```julia
ds = DynamicalSystem(f, u, p) # some concrete implementation
parameters = 0:0.01:1
outputs = zeros(length(parameters))

# Since `DynamicalSystem`s are mutable, we need to copy to parallelize
systems = [deepcopy(ds) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, ds) # we can save 1 copy

Threads.@threads for i in eachindex(parameters)
    system = systems[Threads.threadid()]
    set_parameter!(system, 1, parameters[i])
    outputs[i] = expensive_function(system, args...)
end
```

## Examples

### Iterated map, out of place

Let's make the [Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) as an example.

```@example MAIN
using DynamicalSystemsBase

henon_rule(x, p, n) = SVector(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
u0 = zeros(2)
p0 = [1.4, 0.3]

henon = DeterministicIteratedMap(henon_rule, u0, p0)
```

and get a trajectory

```@example MAIN
X, t = trajectory(henon, 10000; Ttr = 100)
X
```

### Coupled ODEs, in place

Let's make the Lorenz system
[Hénon map](https://en.wikipedia.org/wiki/H%C3%A9non_map) as an example.
The system is small, and therefore should utilize the out of place syntax, but for the case of example, we will use the in-place syntax.
We'll also use a high accuracy solver from OrdinaryDiffEq.jl.

```@example MAIN
using DynamicalSystemsBase
using OrdinaryDiffEq: Vern9

@inbounds function lorenz_rule!(du, u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
    return nothing
end

u0 = [0, 10.0, 0]
p0 = [10, 28, 8/3]
diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)

lorenz = CoupledODEs(lorenz_rule!, u0, p0; diffeq)
```

and get a trajectory

```@example MAIN
X, t = trajectory(lorenz, 1000; Δt = 0.05, Ttr = 10)
X
```