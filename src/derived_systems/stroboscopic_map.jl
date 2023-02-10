export StroboscopicMap, set_period!

###########################################################################################
# Type
###########################################################################################
"""
	StroboscopicMap <: DiscreteTimeDynamicalSystem
	StroboscopicMap(ds::CoupledODEs, period::Real) → smap
	StroboscopicMap(period::Real, f, u0, p = nothing; kwargs...)

A discrete time dynamical system that produces iterations of a time-dependent
(non-autonomous) [`CoupledODEs`](@ref) system exactly over a given `period`.
The second signature first creates a [`CoupledODEs`](@ref) and then calls the first.

`StroboscopicMap` follows the [`DynamicalSystem`](@ref) interface.
In addition, the function `set_period!(smap, period)` is provided,
that sets the period of the system to a new value (as if it was a parameter).
As this system is in discrete time, [`current_time`](@Ref) and [`initial_time`](@ref)
are integers. The initial time is always 0, because `current_time` counts elapsed periods.
Call these functions on the `parent` of `StroboscopicMap` to obtain the
corresponding continuous time.
In contrast, [`reinit!`](@ref) expects `t0` in continuous time.

The convenience constructor
```julia
StroboscopicMap(T::Real, f, u0, p = nothing; diffeq, t0 = 0) → smap
```
is also provided.

See also [`PoincareMap`](@ref).
"""
mutable struct StroboscopicMap{D<:CoupledODEs, TT<:Real} <: DiscreteTimeDynamicalSystem
	ds::D
	period::TT
	t::Int
end

StroboscopicMap(ds::D, T::TT) where {D<:CoupledODEs, TT<:Real} =
StroboscopicMap{D, TT}(ds, T, 0)

StroboscopicMap(T::Real, f, u0::AbstractArray, p = nothing; kwargs...) =
StroboscopicMap(CoupledODEs(f, u0, p; kwargs...), T)

additional_details(smap::StroboscopicMap) = [
    "period" => smap.period,
]
Base.parent(smap::StroboscopicMap) = smap.ds
set_period!(smap::StroboscopicMap, T) = (smap.period = T)

###########################################################################################
# Extend interface
###########################################################################################
for f in (:current_state, :initial_state, :current_parameters, :initial_parameters,
	:dynamic_rule, :set_state!, :(SciMLBase.isinplace), :(StateSpaceSets.dimension))
    @eval $(f)(smap::StroboscopicMap, args...) = $(f)(smap.ds, args...)
end
current_time(smap::StroboscopicMap) = smap.t
initial_time(smap::StroboscopicMap) = 0

function SciMLBase.step!(smap::StroboscopicMap)
	step!(smap.ds, smap.period, true)
	smap.t += 1
	return
end
function SciMLBase.step!(smap::StroboscopicMap, n::Int, stop_at_dt = true)
	step!(smap.ds, n*smap.period, true)
	smap.t += n
	return
end

function SciMLBase.reinit!(smap::StroboscopicMap, u = initial_state(smap);
		p = current_parameters(smap), t0 = initial_time(smap.ds)
	)
	isnothing(u) && return
	smap.t = 0
	reinit!(smap.ds, u; t0, p)
end
