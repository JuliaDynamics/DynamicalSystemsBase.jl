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

A `DynamicalSystem` **represents the time evolution of a state in a state space**.
It mainly encapsulates three things:

1. A state, typically referred to as `u`, with initial value `u0`.
   The space that `u` occupies is the state space of `ds`
   and the length of `u` is the dimension of `ds` (and of the state space).
2. A dynamic rule, typically referred to as `f`, that dictates how the state
   evolves/changes with time when calling the [`step!`](@ref) function.
   `f` is typically a standard Julia function, see the online documentation for examples.
3. A parameter container `p` that parameterizes `f`. `p` can be anything,
   but in general it is recommended to be a type-stable mutable container.

In sort, any set of quantities that change in time can be considered a dynamical system,
however the concrete subtypes of `DynamicalSystem` are much more specific in their scope.
Concrete subtypes typically also contain more information than the above 3 items.

In this scope dynamical systems have a known dynamic rule `f`.
Finite _measured_ or _sampled_ data from a dynamical system
are represented using [`StateSpaceSet`](@ref).
Such data are obtained from the [`trajectory`](@ref) function or
from an experimental measurement of a dynamical system with an unknown dynamic rule.

See also the DynamicalSystems.jl tutorial online for examples making dynamical systems.

## Integration with ModelingToolkit.jl

Dynamical systems that have been constructed from `DEProblem`s that themselves
have been constructed from ModelingToolkit.jl keep a reference to the symbolic
model and all symbolic variables. Accessing a `DynamicalSystem` using symbolic variables
is possible via the functions [`observe_state`](@ref), [`set_state!`](@ref),
[`current_parameter`](@ref) and [`set_parameter!`](@ref).
The referenced MTK model corresponding to the dynamical system can be obtained with
`model = referrenced_sciml_model(ds::DynamicalSystem)`.

See also the DynamicalSystems.jl tutorial online for an example.

!!! warn "ModelingToolkit.jl v9"
    In ModelingToolkit.jl v9 the default `split` behavior of the parameter container
    is `true`. This means that the parameter container is no longer a `Vector{Float64}`
    by default, which means that you cannot use integers to access parameters.
    It is recommended to keep `split = true` (default) and only access
    parameters via their symbolic parameter binding.
    Use `structural_simplify(sys; split = false)` to allow accessing parameters
    with integers again.

## API

The API that `DynamicalSystem` employs is composed of
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
- [`successful_step`](@ref)
- [`referrenced_sciml_model`](@ref)

### API - alter status

- [`reinit!`](@ref)
- [`set_state!`](@ref)
- [`set_parameter!`](@ref)
- [`set_parameters!`](@ref)
"""
abstract type DynamicalSystem end

# Make it broadcastable:
Base.broadcastable(ds::DynamicalSystem) = Ref(ds)

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
    reinit!, set_state!, set_parameter!, set_parameters!, step!, observe_state, referrenced_sciml_model

###########################################################################################
# Symbolic support
###########################################################################################
# Simply extend the `referrenced_sciml_prob` and you have symbolic indexing support!
import SymbolicIndexingInterface
referrenced_sciml_prob(::DynamicalSystem) = nothing

# The rest are all automated!
"""
    referrenced_sciml_model(ds::DynamicalSystem)

Return the ModelingToolkit.jl structurally-simplified model referrenced by `ds`.
Return `nothing` if there is no referrenced model.
"""
referrenced_sciml_model(ds::DynamicalSystem) = referrenced_sciml_model(referrenced_sciml_prob(ds))
referrenced_sciml_model(prob::SciMLBase.DEProblem) = prob.f.sys
referrenced_sciml_model(::Nothing) = nothing

# return true if there is an actual referrenced system
has_referrenced_model(prob::SciMLBase.DEProblem) = has_referrenced_model(referrenced_sciml_model(prob))
has_referrenced_model(::Nothing) = false
has_referrenced_model(::SymbolicIndexingInterface.SymbolCache{Nothing, Nothing, Nothing}) = false
has_referrenced_model(sys) = true

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
See also [`initial_state`](@ref), [`observe_state`](@ref).
"""
current_state(ds::DynamicalSystem) = ds.u

"""
    observe_state(ds::DynamicalSystem, i, u = current_state(ds)) → x::Real

Return the state `u` of `ds` _observed_ at "index" `i`. Possibilities are:

- `i::Int` returns the `i`-th dynamic variable.
- `i::Function` returns `f(current_state(ds))`.
- `i::SymbolLike` returns the value of the corresponding symbolic variable.
   This is valid only for dynamical systems referrencing a ModelingToolkit.jl model
   which also has `i` as one of its listed variables (either uknowns or observed).
   Here `i` can be anything can be anything
   that could index the solution object `sol = ModelingToolkit.solve(...)`,
   such as a `Num` or `Symbol` instance with the name of the symbolic variable.
   In this case, a last fourth optional positional argument `t` defaults to
   `current_time(ds)` and is the time to observe the state at.
- Any symbolic expression involving variables present in the symbolic
  variables tracked by the system, e.g., `i = x^2 - y` with `x, y`
  symbolic variables.

For [`ProjectedDynamicalSystem`](@ref), this function assumes that the
state of the system is the full state space state, not the projected one
(this makes the most sense for allowing MTK-based indexing).

Use [`state_name`](@ref) for an accompanying name.
"""
function observe_state(ds::DynamicalSystem, index, u::AbstractArray = current_state(ds), t = current_time(ds))
    if index isa Function
        return index(u)
    elseif index isa Integer
        return u[index]
    elseif has_referrenced_model(ds)
        prob = referrenced_sciml_prob(ds)
        ugetter = SymbolicIndexingInterface.observed(prob, index)
        p = current_parameters(ds)
        return ugetter(u, p, t)
    else
        throw(ArgumentError("Invalid index to observe state, or if symbolic index, the "*
        "dynamical system does not referrence a ModelingToolkit.jl system."))
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

See also [`initial_parameters`](@ref), [`current_parameter`](@ref), [`set_parameter!`](@ref).
"""
current_parameters(ds::DynamicalSystem) = ds.p

"""
    current_parameter(ds::DynamicalSystem, index [,p])

Return the specific parameter of `ds` corresponding to `index`,
which can be anything given to [`set_parameter!`](@ref).
`p` defaults to [`current_parameters`](@ref) and is the parameter container
to extract the parameter from, which must match layout with its default value.

Use [`parameter_name`](@ref) for an accompanying name.
"""
function current_parameter(ds::DynamicalSystem, index, p = current_parameters(ds))
    prob = referrenced_sciml_prob(ds)
    if !has_referrenced_model(prob)
        return _get_parameter(p, index)
    else # symbolic dispatch
        i = SymbolicIndexingInterface.getp(prob, index)
        return i(p)
    end
end
_get_parameter(p::Union{AbstractArray, AbstractDict}, index) = getindex(p, index)
# Dispatch for composite types as parameter containers
_get_parameter(p, index) = getproperty(p, index)

"""
    initial_parameters(ds::DynamicalSystem) → p0

Return the initial parameter container of `ds`.
This is never mutated and is set when initializing `ds`.
"""
initial_parameters(ds::DynamicalSystem) = ds.p0

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

successful_step(ds::DiscreteTimeDynamicalSystem) = all(x -> (isfinite(x) && !isnan(x)), current_state(ds))

# Generic implementation, most types re-define it as compile-time info
StateSpaceSets.dimension(ds::DynamicalSystem) = length(current_state(ds))

###########################################################################################
# API - altering status of the system
###########################################################################################
"""
    set_state!(ds::DynamicalSystem, u::AbstractArray{Real})

Set the state of `ds` to `u`, which must match dimensionality with that of `ds`.
Also ensure that the change is notified to whatever integration protocol is used.
"""
set_state!(ds, u) = errormsg(ds)

"""
    set_state!(ds::DynamicalSystem, value::Real, i) → u

Set the `i`th variable of `ds` to `value`. The index `i` can be an integer or
a symbolic-like index for systems that reference a ModelingToolkit.jl model.
For example:
```julia
i = :x # or `1` or `only(@variables(x))`
set_state!(ds, 0.5, i)
```

**Warning:** this function should not be used with derivative dynamical systems
such as Poincare/stroboscopic/projected dynamical systems.
Use the method below to manipulate an array and give that to `set_state!`.


    set_state!(u::AbstractArray, value, index, ds::DynamicalSystem)

Modify the given state `u` and leave `ds` untouched.
"""
function set_state!(ds::DynamicalSystem, value::Real, i)
    u = Array(current_state(ds)) # ensure it works for out of place as well!
    u = set_state!(u, value, i, ds)
    set_state!(ds, u)
end

function set_state!(u::AbstractArray, value::Real, i, ds::DynamicalSystem)
    prob = referrenced_sciml_prob(ds)
    if i isa Integer
        u[i] = value
    elseif has_referrenced_model(prob)
        usetter = SymbolicIndexingInterface.setu(prob, i)
        usetter(u, value)
    else
        throw(ArgumentError("Invalid index to set state, or if symbolic index, the "*
        "dynamical system does not referrence a ModelingToolkit.jl system."))
    end
    return u
end

"""
    set_state!(ds::DynamicalSystem, mapping::AbstractDict)

Convenience version of `set_state!` that iteratively calls `set_state!(ds, val, i)`
for all index-value pairs `(i, val)` in `mapping`. This allows you to
partially set only some state variables.
"""
function set_state!(ds::DynamicalSystem, mapping::AbstractDict)
    # ensure we use a mutable vector, so same code works for in-place problems
    # (SymbolicIndexingInterface only works with mutable objects)
    um = Array(copy(current_state(ds)))
    set_state!(um, mapping, ds)
    set_state!(ds, um)
end
function set_state!(um::Array{<:Real}, mapping::AbstractDict, ds::DynamicalSystem)
    for (i, value) in pairs(mapping)
        set_state!(um, value, i, ds)
    end
    return um
end

"""
    set_parameter!(ds::DynamicalSystem, index, value [, p])

Change a parameter of `ds` given the `index` it has in the parameter container
and the `value` to set it to. This function works for any type of parameter container
(array/dictionary/composite types) provided the `index` is appropriate type.

The `index` can be a traditional Julia index (integer for arrays, key for dictionaries,
or symbol for composite types). It can also be a symbolic variable or `Symbol` instance.
This is valid only for dynamical systems referring a ModelingToolkit.jl model
which also has `index` as one of its parameters.

The last optional argument `p` defaults to [`current_parameters`](@ref) and is
the parameter container whose value is changed at the given index.
It must match layout with its default value.
"""
function set_parameter!(ds::DynamicalSystem, index, value, p = current_parameters(ds))
    # internal function is necessary so that we are able to call `u_modified!` for ODEs.
    _set_parameter!(ds::DynamicalSystem, index, value, p)
end
function _set_parameter!(ds::DynamicalSystem, index, value, p = current_parameters(ds))
    prob = referrenced_sciml_prob(ds)
    if !has_referrenced_model(prob)
        if p isa Union{AbstractArray, AbstractDict}
            setindex!(p, value, index)
        else
            setproperty!(p, index, value)
        end
    else
        set! = SymbolicIndexingInterface.setp(prob, index)
        set!(p, value)
    end
    return
end

"""
    set_parameters!(ds::DynamicalSystem, p = initial_parameters(ds))

Set the parameter values in the [`current_parameters`](@ref)`(ds)` to match those in `p`.
This is done as an in-place overwrite by looping over the keys of `p`
hence `p` can be an arbitrary container mapping parameter indices to values
(such as a `Vector{Real}`, `Vector{Pair}`, or `AbstractDict`).

The keys of `p` must be valid keys that can be given to [`set_parameter!`](@ref).
"""
function set_parameters!(ds::DynamicalSystem, p = initial_parameters(ds))
    cp = current_parameters(ds)
    p === cp && return
    iter = p isa Vector ? pairs(p) : p # allows using vector, dict, or vector{pair}.
    for (index, value) in iter
        _set_parameter!(ds, index, value, cp)
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
`ds` first calls this function on it. Besides the new state `u`, you
can also configure the keywords `t0 = initial_time(ds)` and `p = current_parameters(ds)`.

    reinit!(ds::DynamicalSystem, u::AbstractDict; kwargs...) → ds

If `u` is a `AbstractDict` (for partially setting specific state variables in [`set_state!`](@ref)),
then the alterations are done in the state given by the keyword
`reference_state = copy(initial_state(ds))`.

    reinit!(ds, ::Nothing; kwargs...)

This method does nothing and leaves the system as is. This is so that downstream functions
that call `reinit!` can still be used without resetting the system but rather
continuing from its exact current state.
"""
function SciMLBase.reinit!(ds::DynamicalSystem, mapping::AbstractDict;
    reference_state = copy(initial_state(ds)), kwargs...)
    um = Array(reference_state)
    set_state!(um, mapping, ds)
    reinit!(ds, um; kwargs...)
end

SciMLBase.reinit!(ds::DynamicalSystem, ::Nothing; kw...) = ds
# all extensions of `reinit!` in concrete implementaitons
# should only implement `reinit!(ds, ::AbstractArray)` method.