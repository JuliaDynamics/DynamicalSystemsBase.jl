export stroboscopicmap, projectedintegrator


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
* `idxs = 1:length(Dp)`: This vector selects the variables of the system that will define the
  subspace the dynamics will be projected into, with `Dp` the dimension
  of the projected subspace
* `complete_state = zeros(D-Dp)`: This argument allows setting the _remaining_ variables
  of the dynamical system state on each initial condition `u`. It can be either a vector
  of length `D-Dp`, or a function `f(y)` that returns a vector of length `D-Dp` given
  the _projected_ initial condition on the grid `y`.
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
function projectedintegrator(ds::CDS{IIP, S, D};  u0 = get_state(ds),
	idxs = 1:length(get_state(ds)), complete_state = zeros(eltype(get_state(ds)), 0),
	diffeq = NamedTuple()
	) where {IIP, S, D}

	Ds = length(get_state(ds))
    if complete_state isa AbstractVector && (length(complete_state) ≠ Ds-length(idxs))
        error("Vector `complete_state` must have length D-Dp!")
    end

	idxs = SVector(idxs...)
	complete_and_reinit! = ProjectedSystem(complete_state, idxs, length(get_state(ds)))
    get_projected_state = (ds) -> view(get_state(ds), idxs)
	integ = integrator(ds, u0; diffeq)

	return ProjectedIntegratror(integ, complete_and_reinit!, get_projected_state)
end

mutable struct ProjectedIntegratror{I, F, G}
	integ::I
	complete_and_reinit!::F
	get_projected_state::G
end

function DynamicalSystemsBase.step!(psys::ProjectedIntegratror, Δt)
	step!(psys.integ, Δt)
	return psys.get_projected_state(psys.integ)
end
function DynamicalSystemsBase.reinit!(psys::ProjectedIntegratror, u0)
	psys.complete_and_reinit!(psys.integ, u0)
	return
end
function DynamicalSystemsBase.get_state(psys::ProjectedIntegratror)
	return psys.get_projected_state(psys.integ)
end

function Base.show(io::IO, psys::ProjectedIntegratror)
    println(io, "Iterator of the Projected System")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(psys.integ.f.f))
    println(io,  rpad(" Complete state: ", 14),     psys.complete_and_reinit!.complete_state)
end





# Utilities for re-initializing initial conditions on the grid
"""
    ProjectedSystem(complete_state, idxs, D)
Helper struct that completes a state and reinitializes the integrator once called
as a function with arguments `f(integ, y)` with `integ` the initialized dynamical
system integrator and `y` the projected initial condition on the grid.
"""
struct ProjectedSystem{C, Y, R}
    complete_state::C
    u::Vector{Float64} # dummy variable for a state in full state space
    idxs::SVector{Y, Int}
    remidxs::R
end
function ProjectedSystem(complete_state, idxs, D::Int)
    remidxs = setdiff(1:D, idxs)
    remidxs = isempty(remidxs) ? nothing : SVector(remidxs...)
    u = zeros(D)
    if complete_state isa AbstractVector
        @assert eltype(complete_state) <: Number
    end
    return ProjectedSystem(complete_state, u, idxs, remidxs)
end
function (c::ProjectedSystem{<: AbstractVector})(integ, y)
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state
    end
    reinit!(integ, c.u)
end
function (c::ProjectedSystem)(integ, y) # case where `complete_state` is a function
    c.u[c.idxs] .= y
    if !isnothing(c.remidxs)
        c.u[c.remidxs] .= c.complete_state(y)
    end
    reinit!(integ, c.u)
end
