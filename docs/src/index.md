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
isinplace
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
```

## `DeterministicIterativeMap`
```@docs
DeterministicIterativeMap
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
```

## `TangentDynamicalSystem`
```@docs
AnalyticRuleSystem
TangentDynamicalSystem
current_deviations
set_deviations!
orthonormal
```