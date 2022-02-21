export stroboscopicmap

"""
	stroboscopicmap(ds::ContinuousDynamicalSystem, T; kwargs...)  → smap

Return a map (integrator) that produces iterations over a period `T` of the `ds`,
known as a stroboscopic map. See [Integrator API](@ref) for handling integrators.

See also [`poincaremap`](@ref).

## Keyword Arguments
* `u0`: initial state
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.

## Example
```julia
f = 0.27; ω = 0.1
ds = Systems.duffing(zeros(2); ω, f, d = 0.15, β = -1)
smap = stroboscopicmap(ds, 2π/ω; diffeq = (;reltol = 1e-8))
reinit!(smap, [1.0, 1.0])
u = step!(smap)
```
"""
function stroboscopicmap(ds::CDS{IIP, S, D}, T = nothing; u0 = get_state(ds),
	diffeq = NamedTuple()
	) where {IIP, S, D}

	if isnothing(T)
		@warn "T must be defined, taking T=1 as default"
		T =1
	end

	integ = integrator(ds, u0; diffeq)
	return StroboscopicMap(integ, T)
end

mutable struct StroboscopicMap{I, F}
	integ::I
	T::F
end

function DynamicalSystemsBase.step!(smap::StroboscopicMap)
	step!(smap.integ, smap.T, true)
	return smap.integ.u
end
function DynamicalSystemsBase.step!(smap::StroboscopicMap, n)
	step!(smap.integ, n*smap.T, true)
	return smap.integ.u
end
function DynamicalSystemsBase.reinit!(smap::StroboscopicMap, u0)
	reinit!(smap.integ, u0)
	return
end
function DynamicalSystemsBase.get_state(smap::StroboscopicMap)
	return smap.integ.u
end

function Base.show(io::IO, smap::StroboscopicMap)
    println(io, "Iterator of the stroboscopic map")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(smap.integ.f.f))
    println(io,  rpad(" Period: ", 14),     smap.T)
end