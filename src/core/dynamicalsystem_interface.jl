# This file declares the API/interface of `DynamicalSystem`.
export DynamicalSystem, ContinuousTimeDynamicalSystem, DiscreteTimeDynamicalSystem

"""
    DynamicalSystem

`DynamicalSystem` is an abstract supertype encompassing all concrete implementations
of what counts as a "dynamical system" in the DynamicalSystems.jl library.

**_All concrete implementations of `DynamicalSystem`
can be iteratively evolved in time via the [`step!`](@ref) function._**
Hence, most library functions that evolve the system
will mutate its current state and/or parameters. See the documentation online
for implications this has for parallelization.

`DynamicalSystem` is further separated into two abstract types:
`ContinuousTimeDynamicalSystem, DiscreteTimeDynamicalSystem`.
The simplest and most common concrete implementations of a `DynamicalSystem`
are [`DeterministicIteratedMap`](@ref) or [`CoupledODEs`](@ref).

## Description

!!! note
    The documentation of `DynamicalSystem` follows chapter 1 of
    [Nonlinear Dynamics](https://link.springer.com/book/10.1007/978-3-030-91032-7),
    Datseris & Parlitz, Springer 2022.

A `ds::DynamicalSystem` **_representes a flow Φ in a state space_**.
It mainly encapsulates three things:

1. A state, typically referred to as `u`, with initial value `u0`.
   The space that `u` occupies is the state space of `ds`
   and the length of `u` is the dimension of `ds` (and of the state space).
2. A dynamic rule, typically referred to as `f`, that dictates how the state
   evolves/changes with time when calling the [`step!`](@ref) function.
   `f` is typically a standard Julia function, see below.
3. A parameter container `p` that parameterizes `f`. `p` can be anything,
   but in general it is recommended to be a type-stable mutable container.

In sort, any set of quantities that change in time can be considered a dynamical system,
however the concrete subtypes of `DynamicalSystem` are much more specific in their scope.
Concrete subtypes typically also contain more information than the above 3 items.

In this scope dynamical systems have a known dynamic rule `f` defined as a
standard Julia function. _Observed_ or _measured_ data from a dynamical system
are represented using `StateSpaceSet` and are finite.
Such data are obtained from the [`trajectory`](@ref) function or
from an experimental measurement of a dynamical system with an unknown dynamic rule.

## Construction instructions on `f` and `u`

Most of the concrete implementations of `DynamicalSystem`, with the exception of
[`ArbitrarySteppable`](@ref), have two ways of implementing the dynamic rule `f`,
and as a consequence the type of the state `u`. The distinction is done on whether
`f` is defined as an in-place (iip) function or out-of-place (oop) function.

* **oop** : `f` **must** be in the form `f(u, p, t) -> out`
    which means that given a state `u::SVector{<:Real}` and some parameter container
    `p` it returns the output of `f` as an `SVector{<:Real}` (static vector).
* **iip** : `f` **must** be in the form `f!(out, u, p, t)`
    which means that given a state `u::AbstractArray{<:Real}` and some parameter container `p`,
    it writes in-place the output of `f` in `out::AbstractArray{<:Real}`.
    The function **must** return `nothing` as a final statement.

`t` stands for current time in both cases.
**iip** is suggested for systems with high dimension and **oop** for small.
The break-even point is between 10 to 100 dimensions but should be benchmarked
on a case-by-case basis as it depends on the complexity of `f`.

!!! note "Autonomous vs non-autonomous systems"
    Whether the dynamical system is autonomous (`f` doesn't depend on time) or not, it is
    still necessary to include `t` as an argument to `f`. Some algorithms utilize this
    information, some do not, but we prefer to keep a consistent interface either way.
    You can also convert any system to autonomous by making time an additional variable.
    If the system is non-autonomous, its _effective dimensionality_ is `dimension(ds)+1`.

## API

The API that the interface of `DynamicalSystem` employs is
the functions listed below. Once a concrete instance of a subtype of `DynamicalSystem` is
obtained, it can queried or altered with the following functions.

The main use of a concrete dynamical system instance is to provide it to downstream
functions such as `lyapunovspectrum` from ChaosTools.jl or `basins_of_attraction`
from Attractors.jl. A typical user will likely not utilize directly the following API,
unless when developing new algorithm implementations that use dynamical systems.

### API - obtain information

- `ds(t)` with `ds` an instance of `DynamicalSystem`: return the state of `ds` at time `t`.
  For continuous time systems this interpolates and extrapolates,
  while for discrete time systems it only works if `t` is the current time.
- [`current_state`](@ref)
- [`initial_state`](@ref)
- [`observe_state`](@ref)
- [`current_parameters`](@ref)
- [`current_parameter`](@ref)
- [`initial_parameters`](@ref)
- [`isdeterministic`](@ref)
- [`isdiscretetime`](@ref)
- [`dynamic_rule`](@ref)
- [`current_time`](@ref)
- [`initial_time`](@ref)
- [`isinplace`](@ref)
- [`succesful_step`](@ref)

### API - alter status

- [`reinit!`](@ref)
- [`set_state!`](@ref)
- [`set_parameter!`](@ref)
- [`set_parameters!`](@ref)
"""
abstract type DynamicalSystem end

# We utilize nature of time for dispatch; continuous time dispatches to `integ`
# and dispatches for `::DEIntegrator` are defined in `CoupledODEs` file.
"""
    ContinuousTimeDynamicalSystem <: DynamicalSystem

Abstract subtype of `DynamicalSystem` encompassing all continuous time systems.
"""
abstract type ContinuousTimeDynamicalSystem <: DynamicalSystem end

"""
    DiscreteTimeDynamicalSystem <: DynamicalSystem

Abstract subtype of `DynamicalSystem` encompassing all discrete time systems.
"""
abstract type DiscreteTimeDynamicalSystem <: DynamicalSystem end

errormsg(ds) = error("Not yet implemented for dynamical system of type $(nameof(typeof(ds))).")

export current_state, initial_state, current_parameters, current_parameter, initial_parameters, isinplace,
    current_time, initial_time, successful_step, isdeterministic, isdiscretetime, dynamic_rule,
    reinit!, set_state!, set_parameter!, set_parameters!, step!, observe_state

###########################################################################################
# Symbolic support
###########################################################################################
# Simply extend the `referrenced_sciml_sys` and you have symbolic indexing support!
import SymbolicIndexingInterface
# return a tuple of the DEProblem and MTK System and DE Integrator
referrenced_sciml_sys(::DynamicalSystem) = (nothing, nothing, nothing)

# return true if there is an actual referrenced system (instead of nothing)
has_referrenced_sys(::Nothing) = false
has_referrenced_sys(::SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing}) = false
has_referrenced_sys(sys) = true

###########################################################################################
# API - obtaining information from the system
###########################################################################################
function (ds::DiscreteTimeDynamicalSystem)(t::Real)
    if t == current_time(ds)
        return current_state(ds)
    end
    throw(ArgumentError("Cannot interpolate/extrapolate discrete time dynamical systems."))
end

(ds::ContinuousTimeDynamicalSystem)(t::Real) = ds.integ(t)

"""
    current_state(ds::DynamicalSystem) → u::AbstractArray

Return the current state of `ds`. This state is mutated when `ds` is mutated.
"""
current_state(ds::DynamicalSystem) = ds.u

"""
    observe_state(ds::DynamicalSystem, i) → x::Real

Return the current state of `ds` _observed_ at "index" `i`. Possibilities are:

- `i::Int` returns the `i`-th dynamic variable.
- `i::Function` returns `f(current_state(ds))`, which is asserted to be a real number.
- `i::SymbolLike` returns the value of the corresponding symbolic variable.
   This is valid only for dynamical systems referring a ModelingToolkit.jl model
   which also has `i` as one of its listed variables. This can be a formal state variable
   or an "observed" variable according to ModelingToolkit.jl. In short, it can be anything
   that can index the solution object of `sol = solve(...)`. This option becomes available
   when `ModelingToolkit` is loaded.
"""
function observe_state(ds::DynamicalSystem, index)
    u = current_state(ds)
    prob, sys, integ = referrenced_sciml_sys(ds)
    if !has_referrenced_sys(sys)
        T = eltype(u)
        if index isa Int
            return u[index]::T
        elseif index isa Function
            return index(u)::T
        end
    else
        ugetter = SymbolicIndexingInterface.getu(sys, index)
        return ugetter(integ)
    end
end

"""
    initial_state(ds::DynamicalSystem) → u0

Return the initial state of `ds`. This state is never mutated and is set
when initializing `ds`.
"""
initial_state(ds::DynamicalSystem) = ds.u0

"""
    current_parameters(ds::DynamicalSystem) → p

Return the current parameter container of `ds`. This is mutated in functions
that need to evolve `ds` across a parameter range.
"""
current_parameters(ds::DynamicalSystem) = ds.p

"""
    current_parameter(ds::DynamicalSystem, index)

Return the specific parameter corresponding to `index`,
which can be anything given to [`set_parameter!`](@ref).
"""
function current_parameter(ds::DynamicalSystem, index)
    prob, sys, integ = referrenced_sciml_sys(ds)
    if !has_referrenced_sys(sys)
        return _get_parameter(current_parameters(ds), index)
    else # symbolic dispatch
        i = SymbolicIndexingInterface.getp(sys, index)
        return i(prob)
    end
end

# Dispatch for composite types as parameter containers
_get_parameter(p::Union{AbstractArray, AbstractDict}, index) = getindex(p, index)
_get_parameter(p, index) = getproperty(p, index)

"""
    initial_parameters(ds::DynamicalSystem) → p0

Return the initial parameter container of `ds`.
This is never mutated and is set when initializing `ds`.
"""
initial_parameters(ds::DynamicalSystem) = ds.p0
function initial_parameters(ds::DynamicalSystem, index)
    i = parameter_index(ds, index)
    return _get_parameter(initial_parameters(ds), i)
end

"""
    isdeterministic(ds::DynamicalSystem) → true/false

Return `true` if `ds` is deterministic, i.e., the dynamic rule contains no randomness.
This is information deduced from the type of `ds`.
"""
isdeterministic(ds::DynamicalSystem) = true

"""
    isdiscretetime(ds::DynamicalSystem) → true/false

Return `true` if `ds` operates in discrete time, or `false` if it is in continuous time.
This is information deduced from the type of `ds`.
"""
isdiscretetime(ds::ContinuousTimeDynamicalSystem) = false
isdiscretetime(ds::DiscreteTimeDynamicalSystem) = true

"""
    dynamic_rule(ds::DynamicalSystem) → f

Return the dynamic rule of `ds`.
This is never mutated and is set when initializing `ds`.
"""
dynamic_rule(ds::DynamicalSystem) = ds.f

"""
    current_time(ds::DynamicalSystem) → t

Return the current time that `ds` is at. This is mutated when `ds` is evolved.
"""
current_time(ds::DynamicalSystem) = ds.t

"""
    initial_time(ds::DynamicalSystem) → t0

Return the initial time defined for `ds`.
This is never mutated and is set when initializing `ds`.
"""
initial_time(ds::DynamicalSystem) = ds.t0

"""
    isinplace(ds::DynamicalSystem) → true/false

Return `true` if the dynamic rule of `ds` is in-place, i.e., a function mutating the state
in place. If `true`, the state is typically `Array`, if `false`, the state is typically
`SVector`. A front-end user will most likely not care about this information,
but a developer may care.
"""
SciMLBase.isinplace(ds::DynamicalSystem) = errormsg(ds)

"""
    successful_step(ds::DynamicalSystem) -> true/false

Return `true` if the last `step!` call to `ds` was successful, `false` otherwise.
For continuous time systems this uses DifferentialEquations.jl error checking,
for discrete time it checks if any variable is `Inf` or `NaN`.
"""
successful_step(ds::DynamicalSystem) = errormsg(ds)

successful_step(ds::DiscreteTimeDynamicalSystem) = all(x -> (isfinite(x) || !isnan(x)) , current_state(ds))

# Generic implementation, most types re-define it as compile-time info
StateSpaceSets.dimension(ds::DynamicalSystem) = length(current_state(ds))

###########################################################################################
# API - altering status of the system
###########################################################################################
"""
    set_state!(ds::DynamicalSystem, u)

Set the state of `ds` to `u`, which must match dimensionality with that of `ds`.
Also ensure that the change is notified to whatever integration protocol is used.
"""
set_state!(ds, u) = errormsg(ds)

"""
    set_parameter!(ds::DynamicalSystem, index, value)

Change a parameter of `ds` given the `index` it has in the parameter container
and the `value` to set it to. This function works for any type of parameter container
(array/dictionary/composite types) provided the `index` is appropriate type.

The `index` can be a traditional Julia index (integer for arrays, key for dictionaries,
or symbol for composite types). However, it can also be a symbolic variable.
This is valid only for dynamical systems referring a ModelingToolkit.jl model
which also has `index` as one of its parameters.
"""
set_parameter!(ds::DynamicalSystem, args...) = _set_parameter!(ds, current_parameters(ds), args...)

function _set_parameter!(ds::DynamicalSystem, p, index, value)
    prob, sys = referrenced_sciml_sys(ds)
    if !has_referrenced_sys(prob)
        if p isa Union{AbstractArray, AbstractDict}
            setindex!(p, value, index)
        else
            setproperty!(p, index, value)
        end
    else
        i = SymbolicIndexingInterface.setp(sys, index)
        i(prob, value)
    end
    return
end

"""
    set_parameters!(ds::DynamicalSystem, p = initial_parameters(ds))

Set the parameter values in the [`current_parameters`](@ref)`(ds)` to match `p`.
This is done as an in-place overwrite by looping over the keys of `p`.
Hence the keys of `p` must be a subset of the keys of [`current_parameters`](@ref)`(ds)`.
"""
function set_parameters!(ds::DynamicalSystem, p = initial_parameters(ds))
    cp = current_parameters(ds)
    p === cp && return
    for (index, value) in pairs(p)
        _set_parameter!(ds, cp, index, value)
    end
    return
end


###########################################################################################
# API - step and reset
###########################################################################################
"""
    step!(ds::DiscreteTimeDynamicalSystem [, n::Integer]) → ds

Evolve the discrete time dynamical system for 1 or `n` steps.

    step!(ds::ContinuousTimeDynamicalSystem, [, dt::Real [, stop_at_tdt]]) → ds

Evolve the continuous time dynamical system for one integration step.

Alternatively, if a `dt` is given, then progress the integration until
there is a temporal difference `≥ dt` (so, step _at least_ for `dt` time).

When `true` is passed to the optional third argument,
the integration advances for exactly `dt` time.
"""
SciMLBase.step!(ds::DynamicalSystem, args...) = errormsg(ds)

"""
    reinit!(ds::DynamicalSystem, u = initial_state(ds); kwargs...) → ds

Reset the status of `ds`, so that it is as if it has be just initialized
with initial state `u`. Practically every function of the ecosystem that evolves
`ds` first calls this function on it. Besides the new initial state `u`, you
can also configure the keywords `t0 = initial_time(ds)` and `p = current_parameters(ds)`.

Note the default settings: the state and time are the initial,
but the parameters are the current.

The special method `reinit!(ds, ::Nothing; kwargs...)` is also available,
which does nothing and leaves the system as is. This is so that downstream functions
that call `reinit!` can still be used without resetting the system but rather
continuing from its exact current state.
"""
SciMLBase.reinit!(ds::DynamicalSystem, args...; kwargs...) = errormsg(ds)
