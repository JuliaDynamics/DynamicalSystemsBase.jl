export ProjectedDynamicalSystem

##############################################################################################
# Projected API
##############################################################################################
"""
    ProjectedDynamicalSystem <: DynamicalSystem
    ProjectedDynamicalSystem(ds::DynamicalSystem, projection, complete_state)

A dynamical system that represents a projection of an existing `ds` on a (projected) space.

The `projection` defines the projected space. If `projection isa AbstractVector{Int}`,
then the projected space is simply the variable indices that `projection` contains.
Otherwise, `projection` can be an arbitrary function that given the state of the
original system `ds`, returns the state in the projected space. In this case the projected
space can be equal, or even higher-dimensional, than the original.

`complete_state` produces the state for the original system from the projected state.
`complete_state` can always be a function that given the projected state returns a state in
the original space. However, if `projection isa AbstractVector{Int}`, then `complete_state`
can also be a vector that contains the values of the _remaining_ variables of the system,
i.e., those _not_ contained in the projected space. In this case
the projected space needs to be lower-dimensional than the original.

Notice that `ProjectedDynamicalSystem` does not require an invertible projection,
`complete_state` is only used during [`reinit!`](@ref). `ProjectedDynamicalSystem` is
in fact a rather trivial wrapper of `ds` which steps it as normal in the original state
space and only projects as a last step, e.g., during [`current_state`](@ref).

## Examples

Case 1: project 5-dimensional system to its last two dimensions.
```julia
ds = Systems.lorenz96(5)
projection = [4, 5]
complete_state = [0.0, 0.0, 0.0] # completed state just in the plane of last two dimensions
pds = ProjectedDynamicalSystem(ds, projection, complete_state)
reinit!(pds, [0.2, 0.4])
step!(pds)
get_state(pds)
```
Case 2: custom projection to general functions of state.
```julia
ds = Systems.lorenz96(5)
projection(u) = [sum(u), sqrt(u[1]^2 + u[2]^2)]
complete_state(y) = repeat([y[1]/5], 5)
pds = # same as in above example...
```
"""
struct ProjectedDynamicalSystem{P, PD, C, R, D} <: DynamicalSystem
    projection::P
    complete_state::C
    u::Vector{Float64} # dummy variable for a state in full state space
    remidxs::R
	ds::D
end
Base.parent(pds::ProjectedDynamicalSystem) = pds.ds

function ProjectedDynamicalSystem(ds::DynamicalSystem, projection, complete_state)
    u0 = initial_state(ds)
    if projection isa AbstractVector{Int}
        all(1 .≤ projection .≤ dimension(ds)) || error("Dim. of projection must be in 1 ≤ np ≤ dimension(ds)")
        projection = SVector(projection...)
        y = u0[projection]
    else
        projection(u0) isa AbstractVector || error("Projected state must be an AbstracVector")
        y = projection(u0)
    end
    if complete_state isa AbstractVector
        projection isa AbstractVector{Int} || error("Projection vector must be of type Int")
        length(complete_state) + length(projection) == dimension(ds) ||
                                error("Wrong dimensions for complete_state and projection")
        remidxs = setdiff(1:dimension(ds), projection)
        !isempty(remidxs) || error("Error with the indices of the projection")
    else
        length(complete_state(y)) == dimension(ds) ||
                        error("The returned vector of complete_state must equal dimension(ds)")
        remidxs = nothing
    end
    u = zeros(dimension(ds))
	return ProjectedDynamicalSystem{
        typeof(projection), length(y), typeof(complete_state),
        typeof(remidxs), typeof(ds)}(projection, complete_state, u, remidxs, ds)
end

additional_details(pds::ProjectedDynamicalSystem) = [
    "projection" => pds.projection,
    "complete state" => pds.complete_state,
]

###########################################################################################
# Extensions
###########################################################################################
# Everything besides `dimension`, `current/initia_state` and `reinit!` is propagated!
for f in (:(SciMLBase.isinplace), :current_time, :initial_time, :isdiscretetime,
        :current_parameters, :initial_parameters, :isdeterministic, :dynamic_rule,:successful_step
    )
    @eval $(f)(tands::ProjectedDynamicalSystem, args...; kw...) = $(f)(tands.ds, args...; kw...)
end
StateSpaceSets.dimension(::ProjectedDynamicalSystem{P, PD}) where {P, PD} = PD

for f in (:current_state, :initial_state)
    @eval $(f)(pds::ProjectedDynamicalSystem{<:Function}) =
        pds.projection($(f)(pds.ds))
    @eval $(f)(pds::ProjectedDynamicalSystem{<:SVector}) =
        $(f)(pds.ds)[pds.projection]
end
# Extend `step!` just so that it returns the projected system
function SciMLBase.step!(pds::ProjectedDynamicalSystem, args...)
	step!(pds.ds, args...)
	return pds
end

function SciMLBase.reinit!(pds::ProjectedDynamicalSystem{P, D, <:AbstractVector}, y = initial_state(pds); kwargs...) where {P, D}
    isnothing(y) && return(pds)
    u = pds.u
    u[pds.projection] .= y
    u[pds.remidxs] .= pds.complete_state
    reinit!(pds.ds, u; kwargs...)
    return pds
end

function SciMLBase.reinit!(pds::ProjectedDynamicalSystem{P, D, <:Function}, y = initial_state(pds); kwargs...) where {P, D}
    isnothing(y) || reinit!(pds.ds, pds.complete_state(y); kwargs...)
    return pds
end

set_state!(pds::ProjectedDynamicalSystem, u) = reinit!(pds, u)

function (pds::ProjectedDynamicalSystem{P})(t) where {P}
    u = pds.ds(t)
    if P <: Function
        return pds.projection(u)
    elseif P <: SVector
        return u[pds.projection]
    end
end
