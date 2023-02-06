export StroboscopicMap

###########################################################################################
# Type
###########################################################################################
"""
	StroboscopicMap <: DiscreteTimeDynamicalSystem
	StroboscopicMap(ds::CoupledODEs, T::Real) → smap

A discrete time autonomous dynamical system that produces iterations of a time-dependent
(non-autonomous) [`CoupledODEs`](@ref) system exactly over a period `T`.
This is known as a stroboscopic map.
The second signature creates a [`CoupledODEs`](@ref) and calls the first
(with `t0` measured in continuous time).

As this system is in discrete time, [`current_time`](@Ref) and [`initial_time`](@ref)
are integers. The initial time is always 0, because `current_time` counts elapsed periods.
Call these functions on the field `.ds` of `StroboscopicMap` to obtain the
corresponding continuous time.
In contrast, [`reinit!`](@ref) expects `t0` in continuous time.

The convenience constructor
```julia
StroboscopicMap(T::Real, f, u0, p = nothing; diffeq, t0 = 0) → smap
```
is also provided.

See also [`PoincareMap`](@ref).
"""
struct StroboscopicMap{D<:CoupledODEs, TT<:Real} <: DiscreteTimeDynamicalSystem
	ds::D
	T::TT
	t::Base.RefValue{Int}
end

StroboscopicMap(ds::D, T::TT) where {D<:CoupledODEs, TT<:Real} =
StroboscopicMap{D, TT}(ds, T, Ref(0))

StroboscopicMap(T, f, u0, p = nothing; kwargs...) =
StroboscopicMap(CoupledODEs(f, u0, p; kwargs...), T)

additional_details(smap::StroboscopicMap) = [
    "period" => smap.T,
]

###########################################################################################
# Extend interface
###########################################################################################
for f in (:current_state, :initial_state, :current_parameters, :initial_parameters,
	:dynamic_rule, :set_state!, :(SciMLBase.isinplace), :(StateSpaceSets.dimension))
    @eval $(f)(smap::StroboscopicMap, args...) = $(f)(smap.ds, args...)
end
current_time(smap::StroboscopicMap) = smap.t[]
initial_time(smap::StroboscopicMap) = 0

function SciMLBase.step!(smap::StroboscopicMap)
	step!(smap.ds, smap.T, true)
	smap.t[] = smap.t[] + 1
	return
end
function SciMLBase.step!(smap::StroboscopicMap, n::Int, stop_at_dt = true)
	step!(smap.ds, n*smap.T, true)
	smap.t[] = smap.t[] + n
	return
end

function SciMLBase.reinit!(smap::StroboscopicMap, u = initial_state(smap);
		p = current_parameters(smap), t0 = initial_time(smap.ds)
	)
	isnothing(u) && return
	smap.t[] = 0
	reinit!(smap.ds, u; t0, p)
end
