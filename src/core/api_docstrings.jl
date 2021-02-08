#######################################################################################
#                                     Docstrings                                      #
#######################################################################################
"""
    integrator(ds::DynamicalSystem [, u0]; diffeq...) -> integ
Return an integrator object that can be used to evolve a system interactively
using `step!(integ [, Δt])`. Optionally specify an initial state `u0`.

The state of this integrator is a vector.

* `diffeq...` are keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
"""
function integrator end

"""
    get_state(ds::DynamicalSystem)
Return the state of `ds`.

    get_state(integ [, i::Int = 1])
Return the state of the integrator, in the sense of the state of the dynamical system.

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

## Keyword Arguments
* `u0` : Optional different initial state.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

It is *heavily* advised to use the functions [`get_state`](@ref), [`get_deviations`](@ref),
[`set_state!`](@ref), [`set_deviations!`](@ref) to manipulate the integrator.

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
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

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

A `W×D` dataset is returned, with `W = length(t0:dt:T)` with
`t0:dt:T` representing the time vector (*not* returned) and `D` the system dimension.
For discrete systems both `T` and `dt` must be integers.

## Keyword Arguments
* `dt` :  Time step of value output during the solving
  of the continuous system. For discrete systems it must be an integer. Defaults
  to `0.01` for continuous and `1` for discrete.
* `Ttr` : Transient time to evolve the initial state before starting saving states.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  For example `abstol = 1e-9`.  Only valid for continuous systems.
  If you want to specify a solver, do so by using the name `alg`, e.g.:
  `alg = Tsit5(), maxiters = 1000`. This requires you to have been first
  `using OrdinaryDiffEq` to access the solvers. See
  `DynamicalSystemsBase.CDS_KWARGS` for default values.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).
"""
function trajectory end
