export StroboscopicMap

##################################################################################
# Type
##################################################################################
"""
	StroboscopicMap(ds::CoupledODEs, T::Real) <: DynamicalSystem
	StroboscopicMap(T::Real, f, u0, p = nothing; diffeq, t0 = 0)

A discrete time autonomous dynamical system that produces iterations of a time-dependent
(non-autonomous) [`CoupledODEs`](@ref) system exactly over a period `T`.
This is known as a stroboscopic map.
The second signature creates a [`CoupledODEs`](@ref) and calls the first
(with `t0` measured in continuous time).

As this system is in discrete time, [`current_time`](@Ref) and [`initial_time`](@ref)
are integers. The initial time is always 0, because `current_time` counts elapsed periods.
Call these functions on the field `.integ` of `StroboscopicMap` to obtain the
corresponding continuous time.
In contrast, [`reinit!`](@ref) expects `t0` in continuous time.

See also [`PoincareMap`](@ref).
"""
struct StroboscopicMap{D, I, P, TT<:Real} <: DiscreteTimeDynamicalSystem
	integ::I
	p0::P
	T::TT
	t::Base.RefValue{Int}
end

StroboscopicMap(ds::CoupledODEs{D, I, P}, T::TT) where {D, I, P, TT} =
StroboscopicMap{D, I, P, TT}(ds.integ, ds.p0, T, Ref(0))

StroboscopicMap(T, f, u0, p = nothing; kwargs...) =
StroboscopicMap(CoupledODEs(f, u0, p; kwargs...), T)

##################################################################################
# Extend interface
##################################################################################
# Extensions for integrator happen at `CoupledODEs`
for f in (:current_state, :initial_state, :current_parameters, :dynamic_rule,
    :current_time, :set_state!)
    @eval $(f)(ds::StroboscopicMap, args...) = $(f)(ds.integ, args...)
end
SciMLBase.isinplace(ds::StroboscopicMap) = isinplace(ds.integ.f)
StateSpaceSets.dimension(::StroboscopicMap{D}) where {D} = D
current_time(smap::StroboscopicMap) = smap.t[]
initial_time(smap::StroboscopicMap) = 0

function SciMLBase.step!(smap::StroboscopicMap)
	step!(smap.integ, smap.T, true)
	smap.t[] = smap.t[] + 1
	return
end
function SciMLBase.step!(smap::StroboscopicMap, n::Int, stop_at_dt = true)
	for _ in 1:n; step!(smap.integ, smap.T, true); end
	smap.t[] = smap.t[] + n
	return
end

function SciMLBase.reinit!(ds::StroboscopicMap, u = initial_state(ds);
		p0 = current_parameters(ds), t0 = initial_time(ds.integ)
	)
	isnothing(u) && return
	set_parameters!(ds, p0)
	ds.t[] = 0
	reinit!(ds.integ, u; reset_dt = true, t0)
end
