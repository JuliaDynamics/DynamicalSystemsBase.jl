include("hyperplane.jl")
import Roots
export poincaresos, PlaneCrossing, poincaremap, PoincareMap

@deprecate poincaremap PoincareMap

const ROOTS_ALG = Roots.A42()

###########################################################################################
# Type definition
###########################################################################################
"""
	PoincareMap <: DiscreteTimeDynamicalSystem
	PoincareMap(ds::CoupledODEs, plane; kwargs...) → pmap

A discrete time dynamical system that produces iterations over the Poincaré map
of the given continuous time `ds`. This map is defined as the sequence of points on the
Poincaré surface of section, which is defined by the `plane` argument.

See also [`StroboscopicMap`](@ref), [`poincaresos`](@ref), [`produce_orbitdiagram`](@ref).

## Description

The Poincaré surface of section is defined as sequential transversal crossings a trajectory
has with any arbitrary manifold, but for `PoincareMap` the manifold must be a hyperplane.

If the state of `ds` is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the
equation defining a hyperplane is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters of the hyperplane.

In code, `plane` can be either:

* A `Tuple{Int, <: Real}`, like `(j, r)`: the plane is defined
  as when the `j`th variable of the system equals the value `r`.
* A vector of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

`PoincareMap` uses `ds`, higher order interpolation from DifferentialEquations.jl,
and root finding from Roots.jl, to create a high accuracy estimate of the section.

`PoincareMap` follows the [`DynamicalSystems`](@ref) interface with the only
difference that `dimension(pmap) == dimension(ds)`, even though the Poincaré
map is effectively 1 dimension less. [`current_time`](@ref) yields the time the last
crossing of the hyperplane occurred (which is where [`current_state`](@ref) also is).
For the special case of `plane` being a `Tuple{Int, <:Real}`, a special `reinit!` method
is allowed with input state of length `D-1` instead of `D`, i.e., a reduced state already
on the hyperplane that is then converted into the `D` dimensional state.


## Keyword arguments

* `direction = -1`: Only crossings with `sign(direction)` are considered to belong to
  the surface of section. Positive direction means going from less than ``b``
  to greater than ``b``.
* `idxs = 1:dimension(ds)`: Optionally you can choose which variables to save.
  Defaults to the entire state.
* `u0 = current_state(ds)`: Specify an initial state.
* `rootkw = (xrtol = 1e-6, atol = 1e-6)`: A `NamedTuple` of keyword arguments
  passed to `find_zero` from [Roots.jl](https://github.com/JuliaMath/Roots.jl).
* `Tmax = 1e3`: The argument `Tmax` exists so that the integrator can terminate instead
  of being evolved for infinite time, to avoid cases where iteration would continue
  forever for ill-defined hyperplanes or for convergence to fixed points.
  If during one `step!` the system has been evolved for more than `Tmax`,
  then `step!(pmap)` will terminate and error.

## Example

```julia
using DynamicalSystemsBase
ds = Systems.rikitake(zeros(3); μ = 0.47, α = 1.0)
pmap = poincaremap(ds, (3, 0.0))
step!(pmap)
next_state_on_psos = current_state(pmap)
```
"""
mutable struct PoincareMap{I<:ContinuousTimeDynamicalSystem, F, P, R, V} <: DiscreteTimeDynamicalSystem
	ds::I
	plane_distance::F
 	planecrossing::P
	Tmax::Float64
	rootkw::R
	state_on_plane::V
    tcross::Float64
    # These two fields are for setting the state of the pmap from the plane
    # (i.e., given a D-1 dimensional state, create the full D-dimensional state)
    dummy::Vector{Float64}
    diffidxs::Vector{Int}
end

function PoincareMap(
		ds::DS, plane;
        Tmax = 1e3,
	    direction = -1, u0 = current_state(ds),
	    rootkw = (xrtol = 1e-6, atol = 1e-6)
	) where {DS<:ContinuousTimeDynamicalSystem}

    D = dimension(ds)
	check_hyperplane_match(plane, D)
	planecrossing = PlaneCrossing(plane, direction > 0)
	plane_distance = (t) -> planecrossing(ds(t))
    v = recursivecopy(current_state(ds))
    dummy = zeros(D)
    diffidxs = _indices_on_poincare_plane(plane, D)
	pmap = PoincareMap(
        ds, plane_distance, planecrossing, Tmax, rootkw,
        v, current_time(ds), dummy, diffidxs
    )
    step!(pmap)
    return pmap
end

_indices_on_poincare_plane(plane::Tuple, D) = setdiff(1:D, [plane[1]])
_indices_on_poincare_plane(::Vector, D) = collect(1:D-1)


###########################################################################################
# Extensions
###########################################################################################
for f in (:initial_state, :current_parameters, :initial_parameters, :initial_time,
	:dynamic_rule, :set_state!, :(SciMLBase.isinplace), :(StateSpaceSets.dimension))
    @eval $(f)(pmap::PoincareMap, args...) = $(f)(pmap.ds, args...)
end
current_time(pmap::PoincareMap) = pmap.tcross
current_state(pmap::PoincareMap) = pmap.state_on_plane

function SciMLBase.step!(pmap::PoincareMap)
	u, t = poincare_step!(pmap.ds, pmap.plane_distance, pmap.planecrossing, pmap.Tmax, pmap.rootkw)
	if isnothing(u)
		error("Exceeded `Tmax` without crossing the plane.")
	else
		pmap.state_on_plane = u
        pmap.tcross = t
		return pmap
	end
end
SciMLBase.step!(pmap::PoincareMap, n::Int, stop = true) = for _ ∈ 1:n; step!(pmap); end

function SciMLBase.reinit!(pmap::PoincareMap, u0;
        t0 = initial_time(ds), p = current_parameters(ds)
    )
    if length(u0) == dimension(pmap)
	    u = u0
    elseif length(u0) == dimension(pmap) - 1
        u = _recreate_state_from_poincare_plane(pmap, u0)
    else
        error("Dimension of state for poincare map reinit is inappropriate.")
    end
    reinit!(pmap.ds, u; t0, p)
end

function _recreate_state_from_poincare_plane(pmap::PoincareMap, u0)
    plane = pmap.planecrossing.plane
    if plane isa Tuple
        pmap.dummy[pmap.diffidxs] .= u0
        pmap.dummy[plane[1]] = plane[2]
    else
        error("Don't know how to convert state on generic plane into full space.")
    end
    return pmap.dummy
end

###########################################################################################
# Poincare step
###########################################################################################
"""
    poincare_step!(integ, plane_distance, planecrossing, Tmax, rootkw)

Low level function that actually performs the algorithm of finding the next crossing
of the Poincaré surface of section. Return the state and time at the section or `nothing` if
evolved for more than `Tmax` without any crossing.
"""
function poincare_step!(ds, plane_distance, planecrossing, Tmax, rootkw)
    t0 = current_time(ds)
    # Check if initial condition is already on the plane
    side = planecrossing(current_state(ds))
    if side == 0
		dat = current_state(ds)
        step!(ds)
		return dat, t0
    end
    # Otherwise evolve until juuuuuust crossing the plane
    tprev = t0
    while side < 0
        (current_time(ds) - t0) > Tmax && break
        tprev = current_time(ds)
        step!(ds)
        side = planecrossing(current_state(ds))
    end
    while side ≥ 0
        (current_time(ds) - t0) > Tmax && break
        tprev = current_time(ds)
        step!(ds)
        side = planecrossing(current_state(ds))
    end
    # we evolved too long and no crossing, return nothing
    (current_time(ds) - t0) > Tmax && return (nothing, nothing)
    # Else, we're guaranteed to have time window before and after the plane
    time_window = (tprev, current_time(ds))
    tcross = Roots.find_zero(plane_distance, time_window, ROOTS_ALG; rootkw...)
    ucross = ds(tcross)
    return ucross, tcross
end

###########################################################################################
# Poincare surface of section
###########################################################################################
"""
    poincaresos(ds::ContinuousDynamicalSystem, plane, tfinal = 1000.0; kwargs...)

Calculate the Poincaré surface of section[^Tabor1989]
of the given system with the given `plane`.
The system is evolved for total time of `tfinal`.
Return a [`Dataset`](@ref) of the points that are on the surface of section.
See also [`poincaremap`](@ref) for the map version.

If the state of the system is ``\\mathbf{u} = (u_1, \\ldots, u_D)`` then the
equation defining a hyperplane is
```math
a_1u_1 + \\dots + a_Du_D = \\mathbf{a}\\cdot\\mathbf{u}=b
```
where ``\\mathbf{a}, b`` are the parameters of the hyperplane.

In code, `plane` can be either:

* A `Tuple{Int, <: Real}`, like `(j, r)` : the plane is defined
  as when the `j`th variable of the system equals the value `r`.
* A vector of length `D+1`. The first `D` elements of the
  vector correspond to ``\\mathbf{a}`` while the last element is ``b``.

This function uses `ds`, higher order interpolation from DifferentialEquations.jl,
and root finding from Roots.jl, to create a high accuracy estimate of the section.
See also [`produce_orbitdiagram`](@ref).

Notice that `poincaresos` is just a fancy wrapper of initializing a [`poincaremap`](@ref)
and then calling `trajectory` on it.

## Keyword Arguments
* `direction = -1` : Only crossings with `sign(direction)` are considered to belong to
  the surface of section. Positive direction means going from less than ``b``
  to greater than ``b``.
* `idxs = 1:dimension(ds)` : Optionally you can choose which variables to save.
  Defaults to the entire state.
* `Ttr = 0.0` : Transient time to evolve the system before starting
  to compute the PSOS.
* `u0 = get_state(ds)` : Specify an initial state.
* `warning = true` : Throw a warning if the Poincaré section was empty.
* `rootkw = (xrtol = 1e-6, atol = 1e-6)` : A `NamedTuple` of keyword arguments
  passed to `find_zero` from [Roots.jl](https://github.com/JuliaMath/Roots.jl).
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.

[^Tabor1989]:
    M. Tabor, *Chaos and Integrability in Nonlinear Dynamics: An Introduction*,
    §4.1, in pp. 118-126, New York: Wiley (1989)
"""
# function poincaresos(
# 		ds::CDS{IIP, S, D}, plane, tfinal = 1000.0;
# 	    direction = -1, Ttr::Real = 0.0, warning = true, idxs = 1:D, u0 = get_state(ds),
# 	    rootkw = (xrtol = 1e-6, atol = 1e-6), diffeq = NamedTuple(), kwargs...
# 	) where {IIP, S, D}

#     if !isempty(kwargs)
#         @warn DIFFEQ_DEP_WARN
#         diffeq = NamedTuple(kwargs)
#     end

#     check_hyperplane_match(plane, D)
#     integ = integrator(ds, u0; diffeq)
#     i = typeof(idxs) <: Int ? idxs : SVector{length(idxs), Int}(idxs...)
#     planecrossing = PlaneCrossing(plane, direction > 0)
# 	Ttr ≠ 0 && step!(integ, Ttr)
# 	plane_distance = (t) -> planecrossing(integ(t))

# 	data = _poincaresos(integ, plane_distance, planecrossing, tfinal+Ttr, i, rootkw)
#     warning && length(data) == 0 && @warn PSOS_ERROR
#     return Dataset(data)
# end

# # The separation into two functions here exists only to introduce a function barrier
# # for the low level method, to ensure optimization on argments of `poincaremap!`.
# function _poincaresos(integ, plane_distance, planecrossing, tfinal, i, rootkw)
# 	data = _initialize_output(current_state(ds), i)
# 	while current_time(ds) < tfinal
# 		out, t = poincaremap!(integ, plane_distance, planecrossing, tfinal, rootkw)
# 		if !isnothing(out)
#             push!(data, out[i])
#         else
#             break # if we evolved for more than tfinal, we should break anyways.
#         end
# 	end
# 	return data
# end

# _initialize_output(::S, ::Int) where {S} = eltype(S)[]
# _initialize_output(u::S, i::SVector{N, Int}) where {N, S} = typeof(u[i])[]
# function _initialize_output(u, i)
#     error("The variable index when producing the PSOS must be Int or SVector{Int}")
# end

# const PSOS_ERROR = "The Poincaré surface of section did not have any points!"
