include("hyperplane.jl")
import Roots
export PoincareMap, current_crossing_time

@deprecate poincaremap PoincareMap

const ROOTS_ALG = Roots.A42()

###########################################################################################
# Type definition
###########################################################################################
"""
	PoincareMap <: DiscreteTimeDynamicalSystem
	PoincareMap(ds::CoupledODEs, plane; kwargs...) → pmap

A discrete time dynamical system that produces iterations over the Poincaré map[^DatserisParlitz2022]
of the given continuous time `ds`. This map is defined as the sequence of points on the
Poincaré surface of section, which is defined by the `plane` argument.

See also [`StroboscopicMap`](@ref), [`poincaresos`](@ref), [`produce_orbitdiagram`](@ref).

## Keyword arguments

* `direction = -1`: Only crossings with `sign(direction)` are considered to belong to
  the surface of section. Positive direction means going from less than ``b``
  to greater than ``b``.
* `u0 = nothing`: Specify an initial state.
* `rootkw = (xrtol = 1e-6, atol = 1e-6)`: A `NamedTuple` of keyword arguments
  passed to `find_zero` from [Roots.jl](https://github.com/JuliaMath/Roots.jl).
* `Tmax = 1e3`: The argument `Tmax` exists so that the integrator can terminate instead
  of being evolved for infinite time, to avoid cases where iteration would continue
  forever for ill-defined hyperplanes or for convergence to fixed points,
  where the trajectory would never cross again the hyperplane.
  If during one `step!` the system has been evolved for more than `Tmax`,
  then `step!(pmap)` will terminate and error.

## Description

The Poincaré surface of section is defined as sequential transversal crossings a trajectory
has with any arbitrary manifold, but here the manifold must be a hyperplane.
`PoincareMap` iterates over the crossings of the section.

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

`PoincareMap` follows the [`DynamicalSystem`](@ref) interface with the following adjustments:

1. `dimension(pmap) == dimension(ds)`, even though the Poincaré
   map is effectively 1 dimension less.
2. Like [`StroboscopicMap`](@ref) time is discrete and counts the iterations on the
   surface of section. [`initial_time`](@ref) is always `0` and [`current_time`](@ref)
   is current iteration number.
3. A new function [`current_crossing_time`](@ref) returns the real time corresponding
   to the latest crossing of the hyperplane, which is what the [`current_state(ds)`](@ref)
   corresponds to as well.
4. For the special case of `plane` being a `Tuple{Int, <:Real}`, a special `reinit!` method
   is allowed with input state of length `D-1` instead of `D`, i.e., a reduced state already
   on the hyperplane that is then converted into the `D` dimensional state.

## Example

```julia
using DynamicalSystemsBase
ds = Systems.rikitake(zeros(3); μ = 0.47, α = 1.0)
pmap = poincaremap(ds, (3, 0.0))
step!(pmap)
next_state_on_psos = current_state(pmap)
```

[^DatserisParlitz2022]:
    Datseris & Parlitz 2022, _Nonlinear Dynamics: A Concise Introduction Interlaced with Code_,
    [Springer Nature, Undergrad. Lect. Notes In Physics](https://doi.org/10.1007/978-3-030-91032-7)
"""
mutable struct PoincareMap{I<:ContinuousTimeDynamicalSystem, F, P, R, V} <: DiscreteTimeDynamicalSystem
	ds::I
	plane_distance::F
 	planecrossing::P
	Tmax::Float64
	rootkw::R
	state_on_plane::V
    tcross::Float64
	t::Base.RefValue{Int}
    # These two fields are for setting the state of the pmap from the plane
    # (i.e., given a D-1 dimensional state, create the full D-dimensional state)
    dummy::Vector{Float64}
    diffidxs::Vector{Int}
end
Base.parent(pmap::PoincareMap) = pmap.ds

function PoincareMap(
		ds::DS, plane;
        Tmax = 1e3,
	    direction = -1, u0 = nothing,
	    rootkw = (xrtol = 1e-6, atol = 1e-6)
	) where {DS<:ContinuousTimeDynamicalSystem}

    reinit!(ds, u0)
    D = dimension(ds)
	check_hyperplane_match(plane, D)
	planecrossing = PlaneCrossing(plane, direction > 0)
	plane_distance = (t) -> planecrossing(ds(t))
    v = recursivecopy(current_state(ds))
    dummy = zeros(D)
    diffidxs = _indices_on_poincare_plane(plane, D)
	pmap = PoincareMap(
        ds, plane_distance, planecrossing, Tmax, rootkw,
        v, current_time(ds), Ref(0), dummy, diffidxs
    )
    step!(pmap)
    pmap.t[] = 0 # first step is 0
    return pmap
end

_indices_on_poincare_plane(plane::Tuple, D) = setdiff(1:D, [plane[1]])
_indices_on_poincare_plane(::Vector, D) = collect(1:D-1)

additional_details(pmap::PoincareMap) = [
    "hyperplane" => pmap.planecrossing.plane,
    "crossing time" => pmap.tcross,
]

###########################################################################################
# Extensions
###########################################################################################
for f in (:initial_state, :current_parameters, :initial_parameters, :initial_time,
	:dynamic_rule, :set_state!, :(SciMLBase.isinplace), :(StateSpaceSets.dimension))
    @eval $(f)(pmap::PoincareMap, args...) = $(f)(pmap.ds, args...)
end
current_time(pmap::PoincareMap) = pmap.t[]
current_state(pmap::PoincareMap) = pmap.state_on_plane

"""
    current_crossing_time(pmap::PoincareMap) → tcross

Return the time of the latest crossing of the Poincare section.
"""
current_crossing_time(pmap::PoincareMap) = pmap.tcross

function SciMLBase.step!(pmap::PoincareMap)
	u, t = poincare_step!(pmap.ds, pmap.plane_distance, pmap.planecrossing, pmap.Tmax, pmap.rootkw)
	if isnothing(u)
		error("Exceeded `Tmax` without crossing the plane.")
	else
		pmap.state_on_plane = u # this is always a brand new vector
        pmap.tcross = t
        pmap.t[] = pmap.t[] + 1
		return pmap
	end
end
SciMLBase.step!(pmap::PoincareMap, n::Int, s = true) = (for _ ∈ 1:n; step!(pmap); end; pmap)

function SciMLBase.reinit!(pmap::PoincareMap, u0 = initial_state(pmap);
        t0 = initial_time(pmap), p = current_parameters(pmap)
    )
    isnothing(u0) && return pmap
    if length(u0) == dimension(pmap)
	    u = u0
    elseif length(u0) == dimension(pmap) - 1
        u = _recreate_state_from_poincare_plane(pmap, u0)
    else
        error("Dimension of state for `PoincareMap` reinit is inappropriate.")
    end
    reinit!(pmap.ds, u; t0, p)
    step!(pmap) # always step once to reach the PSOS
    pmap.t[] = 0 # first step is 0
    pmap
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
    poincaresos(ds::CoupledODEs, plane, T = 1000.0; kwargs...) → P::Dataset

Return the iterations of `ds` on the Poincaré surface of section with the `plane`,
by evolving `ds` up to a total of `T`.
Return a [`Dataset`](@ref) of the points that are on the surface of section.

This function initializes a [`PoincareMap`](@ref) and steps it until its
[`current_crossing_time`](@ref) exceeds `T`. You can also use [`trajectory`](@ref)
with [`PoincareMap`](@ref) to get a sequence of `N::Int` points instead.

The keywords `Ttr, save_idxs` act as in [`trajectory`](@ref).
See [`PoincareMap`](@ref) for `plane` and all other keywords.
"""
function poincaresos(ds::CoupledODEs, plane, T = 1000.0;
        save_idxs = 1:dimension(ds), Ttr = 0, kwargs...
    )
    pmap = PoincareMap(ds, plane; kwargs...)
    i = typeof(save_idxs) <: Int ? save_idxs : SVector{length(save_idxs), Int}(save_idxs...)
	data = _initialize_output(current_state(pmap), i)
    poincaresos!(data, pmap, i, T, Ttr)
end

function poincaresos!(data, pmap, i, T, Ttr)
    tprev = current_crossing_time(pmap)
    while current_crossing_time(pmap) - tprev < Ttr
        step!(pmap)
    end
    push!(data, current_state(pmap))
    tprev = current_crossing_time(pmap)
    while current_crossing_time(pmap) - tprev < T
        step!(pmap)
        push!(data, current_state(pmap)[i])
    end
    return Dataset(data)
end

_initialize_output(::S, ::Int) where {S} = eltype(S)[]
_initialize_output(u::S, i::SVector{N, Int}) where {N, S} = typeof(u[i])[]
function _initialize_output(u, i)
    error("The variable index when producing the PSOS must be Int or SVector{Int}")
end
