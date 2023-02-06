# DynamicalSystemsBase.jl

```@docs
DynamicalSystemsBase
```

## The `DynamicalSystem` supertype

```@docs
DynamicalSystem
```

## API for `DynamicalSystem`

### Information

```@docs
- [`current_state`](@ref)
- [`initial_state`](@ref)
- [`current_parameters`](@ref)
- [`initial_parameters`](@ref)
- [`isdeterministic`](@ref)
- [`isdiscretetime`](@ref)
- [`dynamic_rule`](@ref)
- [`current_time`](@ref)
- [`initial_time`](@ref)
- [`isinplace`](@ref)
```

### Alteration
```@docs
- [`reinit!`](@ref)
- [`set_state!`](@ref)
- [`set_parameter!`](@ref)
- [`set_parameters!`](@ref)
```
