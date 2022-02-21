export projectedintegrator
#####################################################################################
# Projected API
#####################################################################################
"""
    projected_integrator(ds::DynamicalSystem, projection, complete_state; kwargs...) → integ

Return an integrator that produces iterations of the dynamical system `ds` on a
projected subspace. See [Integrator API](@ref) for handling integrators.

The `projection` defines the projected space. If `projection isa AbstractVector{Int}`,
then the projected space is simply the variable indices that `projection` contains.
Otherwise, `projection` can be an arbitrary function that given the state of the
original system, return the state of the projected system.

`complete_state` is a function that when given as input the state of the projected system
it produces the state for the original system. This is necessary as the actual integration
happens in the full space, and the projected space is only what is returned with e.g.,
[`get_state`](@ref).

## Keyword Arguments
* `u0`: initial state
* `diffeq` is a `NamedTuple` (or `Dict`) of keyword arguments propagated into
  `init` of DifferentialEquations.jl.

## Examples
```julia
ds = Systems.lorenz()
psys = projectedsystem(ds, 0.1; idxs = 1:2, complete_state=[0.0], diffeq = (;reltol = 1e-8))
reinit!(psys,[1., 1.])
step!(psys)
u = get_state(psys)
```
"""
function projected_integrator(ds::CDS{IIP, S, D};  u0 = get_state(ds),
	idxs = 1:length(get_state(ds)), complete_state = zeros(eltype(get_state(ds)), 0),
	projection = nothing, diffeq = NamedTuple()
	) where {IIP, S, D}

	Ds = length(get_state(ds))
    if complete_state isa AbstractVector && (length(complete_state) ≠ Ds-length(idxs))
        error("Vector `complete_state` must have length D-Dp!")
    end

	idxs = SVector(idxs...)
	complete_and_reinit! = ProjectedSystem(complete_state, idxs, length(get_state(ds)))
	if isnothing(projection)
    	get_projected_state = (ds) -> view(get_state(ds), idxs)
	elseif projection isa Function
		get_projected_state = (ds) -> projection(get_state(ds))
	end
	integ = integrator(ds, u0; diffeq)

	return ProjectedIntegratror(integ, complete_and_reinit!, get_projected_state)
end

struct ProjectedIntegratror{I, F, G, Dp}
	integ::I
	complete_and_reinit!::F
	get_projected_state::G
end

get_state(psys::ProjectedIntegratror) = psys.get_projected_state(psys.integ)
function DynamicalSystemsBase.step!(psys::ProjectedIntegratror, args...)
	step!(psys.integ, args...)
	return
end
function DynamicalSystemsBase.reinit!(psys::ProjectedIntegratror, u0)
	psys.complete_and_reinit!(psys.integ, u0)
	return
end
function DynamicalSystemsBase.get_state(psys::ProjectedIntegratror)
	return psys.get_projected_state(psys.integ)
end
function DynamicalSystemsBase.get_state(psys::ProjectedIntegratror)
	return psys.get_projected_state(psys.integ)
end

function Base.show(io::IO, psys::ProjectedIntegratror)
    println(io, "Iterator of the Projected System")
    println(io,  rpad(" rule f: ", 14),     DynamicalSystemsBase.eomstring(psys.integ.f.f))
    println(io,  rpad(" Complete state: ", 14),     psys.complete_and_reinit!.complete_state)
end



#####################################################################################
# Complete and Reinit code
#####################################################################################
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
