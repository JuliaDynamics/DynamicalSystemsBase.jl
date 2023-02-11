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
Dataset
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

Since `DynamicalSystem`s are mutable, one needs to copy them before parallelizing, to avoid having to deal with complicated race conditions etc. The simplest way is with `deepcopy`. Here is an example block that shows how to parallelize calling some expensive function (e.g., calculating the Lyapunov exponent) over a parameter range:

```julia
ds = DynamicalSystem(f, u, p)
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
