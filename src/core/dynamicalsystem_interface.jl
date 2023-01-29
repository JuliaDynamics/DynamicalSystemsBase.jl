# This file declares the API/interface of `DynamicalSystem`.

"""
    DynamicalSystem

`DynamicalSystem` is an abstract supertype encompassing all concrete implementations
of what counts as a "dynamical system" in the DynamicalSystems.jl library.
The alias `DS` is sometimes used in the documentation instead of `DynamicalSystem`.

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

The API that the interface of `DynamicalSystem` employs is
the functions listed below. Once a concrete instance of a subtype of `DynamicalSystem` is
obtained, it can quieried or altered with the following functions.

The main use of a concrete dynamical system instance is to provide it to downstream
functions such as `lyapunovspectrum` from ChaosTools.jl or `basins_of_attraction`
from Attractors.jl. so a typical user will likely not utilize directly the following API,
unless when developing new algorithm implementations that use dynamical systems.

### API - information

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

### API - alter status

- [`reinit!`](@ref)
- [`set_state!`](@ref)
- [`set_parameter!`](@ref)
"""
abstract type DynamicalSystem end
const DS = DynamicalSystem

errormsg(ds) = "Function not implemented for dynamical system of type $(nameof(typeof(ds)))"

##################################################################################
# API - information
##################################################################################
"""
    current_state(ds::DynamicalSystem) → u

Return the current state of `ds`. This state is mutated when `ds` is mutated.
"""
current_state(ds::DynamicalSystem) = errormsg(ds)

"""
    initial_state(ds::DynamicalSystem) → u0

Return the initial state of `ds`. This state is never mutated and is set
when initializing `ds`.
"""
initial_state(ds::DynamicalSystem) = errormsg(ds)

"""
    current_parameters(ds::DynamicalSystem) → p

Return the current parameter container of `ds`. This is mutated in functions
that need to evolve `ds` across a parameter range.
"""
current_parameters(ds::DynamicalSystem) = errormsg(ds)

"""
    initial_parameters(ds::DynamicalSystem) → p0

Return the initial parameter container of `ds`.
This is never mutated and is set when initializing `ds`.
"""
initial_parameters(ds::DynamicalSystem) = errormsg(ds)

"""
    isdeterministic(ds::DynamicalSystem) → true/false

Return `true` if `ds` is deterministic, i.e., the dynamic rule contains no randomness.
This is information deduced from the type of `ds`.
"""
isdeterministic(ds::DynamicalSystem) = errormsg(ds)

"""
    isdiscretetime(ds::DynamicalSystem) → true/false

Return `true` if `ds` operates in discrete time, or `false` if it is in continuous time.
This is information deduced from the type of `ds`.
"""
isdiscretetime(ds::DynamicalSystem) = errormsg(ds)

"""
    dynamic_rule(ds::DynamicalSystem) → f

Return the dynamic rule of `ds`.
This is never mutated and is set when initializing `ds`.
"""
dynamic_rule(ds::DynamicalSystem) = errormsg(ds)

"""
    current_time(ds::DynamicalSystem) → t

Return the current time that `ds` is at. This is mutated when `ds` is evolved.
"""
current_time(ds::DynamicalSystem) = errormsg(ds)

"""
    initial_time(ds::DynamicalSystem) → t0

Return the initial time defined for `ds`.
This is never mutated and is set when initializing `ds`.
"""
initial_time(ds::DynamicalSystem) = errormsg(ds)

"""
    isinplace(ds::DynamicalSystem) → true/false

Return `true` if the dynamic rule of `ds` is in-place, i.e., a function mutating the state
in place. If `true`, the state is typically `Array`, if `false`, the state is typically
`SVector`. A front-end user will most likely not care about this information,
but a developer may care.
"""
SciMLBase.isinplace(ds::DynamicalSystem) = errormsg(ds)

##################################################################################
# API - alter status
##################################################################################

"""
    set_state!(ds::DynamicalSystem, u)

Set the state of `ds` to `u`, which must match dimensionality with that of `ds`.
Also ensure that the change is notified to whatever integratio protocol is used.
"""
set_state!(ds, u) = errormsg(ds)
