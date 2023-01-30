export StroboscopicMap

##################################################################################
# Type
##################################################################################
"""
	StroboscopicMap(ds::CoupledODEs, T::Real) <: DynamicalSystem
	StroboscopicMap(T::Real, f, u0, p = nothing; diffeq, t0 = nothing)

A discrete time dynamical system that produces iterations of a time-dependent
(non-autonomous) [`CoupledODEs`](@ref) system exactly over a period `T`.
This is known as a stroboscopic map.
The second signature creates a [`CoupledODEs`](@ref) and calls the first.

See also [`poincaremap`](@ref).
"""
struct StroboscopicMap{D, I, P, TT<:Real} <: DiscreteTimeDynamicalSystem
	integ::I
	p0::P
	T::TT
end

StroboscopicMap(ds::CoupledODEs{D, I, P}, T::TT) where {D, I, P, TT} =
StroboscopicMap{D, I, P, TT}(ds.integ, ds.p0, T)

StroboscopicMap(T, f, u0, p = nothing; kwargs...) =
StroboscopicMap(CoupledODEs(f, u0, p; kwargs...), T)

##################################################################################
# Extend interface
##################################################################################
for f in (:current_state, :initial_state, :current_parameters, :dynamic_rule,
    :current_time, :initial_time, :set_state!)
    @eval $(f)(ds::StroboscopicMap, args...) = $(f)(ds.integ, args...)
end
SciMLBase.isinplace(ds::StroboscopicMap) = isinplace(ds.integ.f)
StateSpaceSets.dimension(::StroboscopicMap{D}) where {D} = D

function SciMLBase.step!(smap::StroboscopicMap)
	step!(smap.integ, smap.T, true)
	return
end
function SciMLBase.step!(smap::StroboscopicMap, n::Int)
	for _ in 1:n; step!(smap.integ, smap.T, true); end
	return
end

function SciMLBase.reinit!(ds::StroboscopicMap, u = initial_state(ds);
		p0 = current_parameters(ds), t0 = initial_time(ds)
	)
	isnothing(u) && return
	set_parameters!(ds, p0)
	reinit!(ds.integ, u; reset_dt = true, t0)
end
