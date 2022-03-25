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
u = step!(smap, 4) # step 4 iterations forward
```
"""
function stroboscopicmap(ds::CDS, T; u0 = get_state(ds), diffeq = NamedTuple())
	integ = integrator(ds, u0; diffeq)
	return StroboscopicMap{typeof(integ), dimension(ds), typeof(T)}(integ, T)
end
struct StroboscopicMap{I, D, F} <: GeneralizedDynamicalSystem
	integ::I
	T::F
end
isdiscretetime(::StroboscopicMap) = true
DelayEmbeddings.dimension(::StroboscopicMap{I, D}) where {I, D} = D

integrator(p::StroboscopicMap) = p
function step!(smap::StroboscopicMap)
	step!(smap.integ, smap.T, true)
	return smap.integ.u
end
function step!(smap::StroboscopicMap, n::Int)
	for k in 1:n; step!(smap.integ, smap.T, true); end
	return smap.integ.u
end
function reinit!(smap::StroboscopicMap, u0)
	reinit!(smap.integ, u0)
	return
end
function get_state(smap::StroboscopicMap)
	return smap.integ.u
end

function Base.show(io::IO, smap::StroboscopicMap)
    println(io, "Iterator of the stroboscopic map")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(smap.integ.f.f))
    println(io,  rpad(" Period: ", 14),     smap.T)
end

current_time(smap::StroboscopicMap) = current_time(smap.integ)
function (smap::StroboscopicMap)(t)
    error("Time interpolation is not possible for a stroboscopic map!")
end
integrator(pinteg::StroboscopicMap, args...; kwargs...) = pinteg
