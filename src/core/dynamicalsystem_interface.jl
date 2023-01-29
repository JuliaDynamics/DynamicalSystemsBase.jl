# This file declares the API/interface of `DynamicalSystem`.

"""
    DynamicalSystem

`DynamicalSystem` is an abstract supertype encompassing all concrete implementations
of what counts as a "dynamical system" in the DynamicalSystems.jl library.

**_All concrete implementations of `DynamicalSystem` are mutable objects that
can be iteratively evolved in time via the [`step!`](@ref) function._**
Since they are mutable, most library functions that need to evolve the system
will mutate its current state and/or parameters. See the documentation online
for implications this has on e.g., parallelization.

## Description

!!! note
    The documentation of `DynamicalSystem` follows chapter 1 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

A `ds::DynamicalSystem` **_representes a flow Φ in a state space_**.
It mainly encapsulates three things:

1. A state, typically referred to as `u`.
   The space that `u` occupies is the state space of `ds`
   and the size of `u` is the dimension of `ds` (and of the state space).
2. A dynamic rule, typically referred to as `f`, that dictates how the state
   evolves/changes with time when calling the [`step!`](@ref) function.
3. A parameter container that parameterizes `f`.

In sort, any set of quantities that change in time can be considered a dynamical system,
however the concrete subtypes of `DynamicalSystem` are much more specific in their scope.
Concrete subtypes typically also contain more information than the above 3 items.

A dynamical system typically has a known evolution rule defined as a standard Julia
function. `Dataset` is used to encode finite data observed from a dynamical system.

## API

The API that the interface of `DynamicalSystems` employs is
the functions listed below. Once a concrete instance of a subtype of `DynamicalSystem` is
obtained, it can quieried with the following functions. Of course, the main use
of such a concrete instance is to provide it to downstream packages such as
`lyapunovspectrum` from ChaosTools.jl or `basins_of_attraction` from Attractors.jl,
so a typical user will likely not utilize directly the following API, unless when
developing new algorithm implementations that use dynamical systems.

### API - information

- [`current_state`](@ref)
- [`initial_state`](@ref)
- [`current_parameters`](@ref)
- [`initial_parameters`](@ref)
- [`dynamic_rule`](@ref)
- [`get_parameters`](@ref)
- [`isdeterministic`](@ref)
- [`isdiscretetime`](@ref)
- [`isinplace`](@ref)

### API - alter status

- [`reinit!`](@ref)
- [`set_state!`](@ref)
- [`set_parameter!`](@ref)
"""
abstract type DynamicalSystem end

