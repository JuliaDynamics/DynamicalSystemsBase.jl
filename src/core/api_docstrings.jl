#######################################################################################
#                                     Docstrings                                      #
#######################################################################################
"""
    integrator(ds::DynamicalSystem [, u0]; diffeq) -> integ
Return an integrator object that can be used to evolve a system interactively
using `step!(integ [, Δt])`. Optionally specify an initial state `u0`.

The state of this integrator is a vector.

* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
"""
function integrator end

"""
    get_state(ds::DynamicalSystem)
Return the state of `ds`.

    get_state(integ [, i::Int = 1])
Return the state of the integrator, in the sense of the state of the dynamical system.
This function works for any integrator of the library.

If the integrator is a [`parallel_integrator`](@ref), passing `i` will return
the `i`-th state. The function also correctly returns the true state of the system
for tangent integrators.
"""
get_state(integ) = integ.u

"""
    set_state!(integ, u [, i::Int = 1])
Set the state of the integrator to `u`, in the sense of the state of the
dynamical system. Works for any integrator (normal, tangent, parallel).

For parallel integrator, you can choose which state to set (using `i`).

Automatically does `u_modified!(integ, true)`.
"""
set_state!(integ, u) = (integ.u = u; u_modified!(integ, true))

"""
    tangent_integrator(ds::DynamicalSystem, Q0 | k::Int; kwargs...)
Return an integrator object that evolves in parallel both the system as well
as deviation vectors living on the tangent space, also called linearized space.

`Q0` is a *matrix* whose columns are initial values for deviation vectors. If
instead of a matrix `Q0` an integer `k` is given, then `k` random orthonormal
vectors are choosen as initial conditions.

It is *heavily* advised to use the functions [`get_state`](@ref), [`get_deviations`](@ref),
[`set_state!`](@ref), [`set_deviations!`](@ref) to manipulate the integrator.

## Keyword Arguments
* `u0` : Optional different initial state.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.


## Description

If ``J`` is the jacobian of the system then the *tangent dynamics* are
the equations that evolve in parallel the system as well as
a deviation vector (or matrix) ``w``:
```math
\\begin{aligned}
\\dot{u} &= f(u, p, t) \\\\
\\dot{w} &= J(u, p, t) \\times w
\\end{aligned}
```
with ``f`` being the equations of motion and ``u`` the system state.
Similar equations hold for the discrete case.
"""
function tangent_integrator end

"""
    get_deviations(tang_integ)
Return the deviation vectors of the [`tangent_integrator`](@ref) in a form
of a matrix with columns the vectors.
"""
function get_deviations end

"""
    set_deviations!(tang_integ, Q)
Set the deviation vectors of the [`tangent_integrator`](@ref) to `Q`, which must
be a matrix with each column being a deviation vector.

Automatically does `u_modified!(tang_integ, true)`.
"""
function set_deviations! end

"""
    parallel_integrator(ds::DynamicalSystem, states; kwargs...)
Return an integrator object that can be used to evolve many `states` of
a system in parallel at the *exact same times*, using `step!(integ [, Δt])`.

`states` are expected as vectors of vectors.

## Keyword Arguments
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

It is *heavily* advised to use the functions [`get_state`](@ref) and
[`set_state!`](@ref) to manipulate the integrator. Provide `i` as a second
argument to change the `i`-th state.
"""
function parallel_integrator end

"""
    trajectory(ds::DynamicalSystem, T [, u]; kwargs...) -> dataset

Return a dataset that will contain the trajectory of the system,
after evolving it for total time `T`, optionally starting from state `u`.
See [`Dataset`](@ref) for info on how to use this object.

The time vector is `t = (t0+Ttr):Δt:(t0+Ttr+T)` and is not returned
(`t0` is the starting time of `ds` which is by default `0`).

## Keyword Arguments
* `Δt` :  Time step of value output. For discrete systems it must be an integer.
  Defaults to `0.01` for continuous and `1` for discrete.
* `Ttr=0` : Transient time to evolve the initial state before starting saving states.
* `save_idxs::AbstractVector{Int}` : Which variables to output in the dataset (by default all).
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl. Only valid for continuous systems, see below.


## DifferentialEquations.jl keyword arguments
Continuous dynamical systems are evolved via the solvers of DifferentialEquations.jl.
Functions in DynamicalSystems.jl allow providing options to these solvers via the
`diffeq` keyword. For example you could use `diffeq = (abstol = 1e-9, reltol = 1e-9)`.
If you want to specify a solver, do so by using the keyword `alg`, e.g.:
`diffeq = (alg = Tsit5(), maxiters = 100000)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers. See the
`CDS_KWARGS` variable for the default values we use.
Notice that `diffeq` keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

Keep in mind that the default solver is `SimpleATsit5`, which only supports
adaptive time-stepping. Use `(alg = SimpleTsit5(), dt = your_step_size)` as keywords
for a non-adaptive time stepping solver, which is mandatory in some situations
(such as e.g., calculating [`basins_of_attraction`](@ref) of a stroboscopic map).
You can also choose any other solver except `SimpleATsit5`, such as `Tsit5`, as long
as you turn off adaptive stepping, e.g.
`(alg = Tsit5(), adaptive = false, dt = your_step_size)`.
"""
function trajectory end


"""
    reinit!(integ [, state])
Re-initialize the integrator to the new given `state`. The `state` type should match
the integrator type. Specifically:

* `integrator, stroboscopicmap, poincaremap`: it needs to be a vector of size the same
  as the dimension of the dynamical system that produced the integrator.
* `projected_integrator`: it needs to be a vector of same size as the projected space.
* `parallel_integrator`: it needs to be a vector of vectors of size the same as the
  dimension of the dynamical system that produced the integrator.
* `tangent_integrator`: there the special signature `reinit!(integ, state, Q0::AbstractMatrix)`
  should be used, with `Q0` containing the deviation vectors. Alternatively, use
  [`set_deviations!`](@ref) if you only want to change the deviation vectors.
"""
function SciMLBase.reinit!(ds::DynamicalSystem) end


"""
    step!(discrete_integ [, dt::Int])
Progress the discrete-time integrator for 1 or `dt` steps.

    step!(continuous_integ, [, dt [, stop_at_tdt]])
Progress the continuous-time integrator for one step.

Alternative, if a `dt` is given, then `step!` the integrator until
there is a temporal difference `≥ dt` (so, step _at least_ for `dt` time).

When `true` is passed to the optional third argument,
the integrator advances for exactly `dt` time.
"""
function SciMLBase.step!(ds::DynamicalSystem) end

# Util functions for `trajectory`
svector_access(::Nothing) = nothing
svector_access(x::AbstractArray) = SVector{length(x), Int}(x...)
svector_access(x::Int) = SVector{1, Int}(x)
obtain_access(u, ::Nothing) = u
obtain_access(u, i::SVector) = u[i]
