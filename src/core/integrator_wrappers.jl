export stroboscopicmap, projectedsystem


"""
	stroboscopicmap(ds::ContinuousDynamicalSystem, T; kwargs...)  → smap

Return a map (integrator) that produces iterations over a period T of the `ds`.

You can progress the map one step on the section by calling `step!(smap)`,
which also returns the next state. You can also set the integrator to start from a new
state `u` by using `reinit!(smap, u)` and then calling `step!` as normally.

## Keyword Arguments
* `u0`: initial state
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.


## Example
```julia
F = 0.27; ω = 0.1;  # smooth boundary
ds = Systems.duffing(zeros(2), ω = ω, f = F, d = 0.15, β = -1)
smap = stroboscopicmap(ds, 2*pi/ω; diffeq = (;reltol = 1e-8))
reinit!(smap,[1., 1.])
step!(smap)
u = get_state(smap)
```
"""
function stroboscopicmap(ds::CDS{IIP, S, D}, T = nothing; u0 = get_state(ds),
	diffeq = NamedTuple(), kwargs...
	) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

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
function DynamicalSystemsBase.reinit!(smap::StroboscopicMap, u0)
	reinit!(smap.integ, u0)
	return
end
function DynamicalSystemsBase.get_state(smap::StroboscopicMap)
	return smap.integ.u
end

function Base.show(io::IO, smap::StroboscopicMap)
    println(io, "Iterator of the Stroboscopic map")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(smap.integ.f.f))
    println(io,  rpad(" Period: ", 14),     smap.T)
end



"""
	projectedsystem(ds::ContinuousDynamicalSystem, Δt; kwargs...)  → psys

Returns an integrator that produces iterations of the dynamical system `ds` on a
projected subspace.

You can progress the map one step on the section by calling `step!(psys)`,
which also returns the next state. You can also set the integrator to start from a new
state `u` by using `reinit!(psys, u)` and then calling `step!` as normally.

## Keyword Arguments
* `u0`: initial state
* `idxs = 1:length(D)`: This vector selects the variables of the system that will define the
  subspace the dynamics will be projected into.
* `complete_state = zeros(D-Dp)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`, with `Dp` the dimension
  of the projected subspace. It can be either a vector of length `D-Dp`, or a function `f(y)` that
  returns a vector of length `D-Dp` given the _projected_ initial condition on the grid `y`.
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.


## Example
```julia
ds = Systems.lorenz_iip()
psys = projectedsystem(ds, 0.1; idxs = 1:2, complete_state=[0.0], diffeq = (;reltol = 1e-8))
reinit!(psys,[1., 1.])
step!(psys)
u = get_state(psys)
```
"""
function projectedsystem(ds::CDS{IIP, S, D}, Δt = nothing;  u0 = get_state(ds),
	idxs = 1:length(get_state(ds)), complete_state = zeros(eltype(get_state(ds)), 0),
	diffeq = NamedTuple(), kwargs...
	) where {IIP, S, D}

    if !isempty(kwargs)
        @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

	if isnothing(Δt)
		@warn "Δt must be defined, taking Δt=0.01 as default"
		Δt = 0.01
	end

	Ds = length(get_state(ds))
    if complete_state isa AbstractVector && (length(complete_state) ≠ Ds-length(idxs))
        error("Vector `complete_state` must have length D-Dg!")
    end

	idxs = SVector(idxs...)
	complete_and_reinit! = CompleteAndReinit(complete_state, idxs, length(get_state(ds)))
    get_projected_state = (ds) -> view(get_state(ds), idxs)
	integ = integrator(ds, u0; diffeq)

	return ProjectedSystem(integ, Δt, complete_and_reinit!, get_projected_state)
end

mutable struct ProjectedSystem{I, T, F, G}
	integ::I
	Δt::T
	complete_and_reinit!::F
	get_projected_state::G
end

function DynamicalSystemsBase.step!(psys::ProjectedSystem)
	step!(psys.integ, psys.Δt)
	return psys.get_projected_state(psys.integ)
end
function DynamicalSystemsBase.reinit!(psys::ProjectedSystem, u0)
	psys.complete_and_reinit!(psys.integ, u0)
	return
end
function DynamicalSystemsBase.get_state(psys::ProjectedSystem)
	return psys.get_projected_state(psys.integ)
end

function Base.show(io::IO, psys::ProjectedSystem)
    println(io, "Iterator of the Projected System")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(psys.integ.f.f))
    #println(io,  rpad(" Period: ", 14),     psys.T)
end
