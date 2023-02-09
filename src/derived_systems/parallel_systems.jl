# Parallel integration is a bit special;
# For `CoreDynamicalSystem` a dedicated structure exists that uses the existing
# integrators with a vector of vectors.
# For all discrete time systems another structure exists that deepcopies the systems.
# And for all continuous time systems another structure exists.
# TODO: Continous time utilizing `step!(integ, dt, true)` and requiring a `dt`.
export ParallelDynamicalSystem, current_states, initial_states

"""
    ParallelDynamicalSystem <: DynamicalSystem
    ParallelDynamicalSystem(ds::DynamicalSystem, states::Vector{<:AbstractArray})

A struct that evolves several `states` of a given dynamical system in parallel
**at exactly the same times**. Useful when wanting to evolve several different trajectories
of the same system while ensuring that they share parameters and time vector.

This struct follows the [`DynamicalSystem`](@ref) interface with the following adjustments:

- The function [`current_state`](@ref) is called as `current_state(pds, i::Int = 1)`
  which returns the `i`th state. Same for [`initial_state`](@ref).
- Similarly, [`set_state!`](@ref) obtains a second argument `i::Int = 1` to
  set the `i`-th state.
- [`current_states`](@ref) and [`initial_states`](@ref) can be used to get
  all parallel states.
- [`reinit!`](@ref) takes in a vector of states (like `states`) for `u`.
"""
abstract type ParallelDynamicalSystem <: DynamicalSystem end

# Generic interface that doesn't depend on implementation
isinplace(::ParallelDynamicalSystem) = true
current_state(pdsa::ParallelDynamicalSystem, i::Int = 1) = current_states(pdsa)[i]
initial_state(pdsa::ParallelDynamicalSystem, i::Int = 1) = initial_states(pdsa)[i]

###########################################################################################
# Analytically knwon rule: creation
###########################################################################################
# We don't parameterize the dimension because it does not need to be known
# at compile time given the usage of the integrator.
# It uses the generic `DynamicalSystem` dispatch.
# But we do need a special extra parameter that checks if the system
# is ODE _and_ inplace, because we need a special matrix state in this case
struct ParallelDynamicalSystemAnalytic{D, M} <: ParallelDynamicalSystem
    ds::D      # standard dynamical system but with rule the parallel dynamics
    original_f # no type parameterization here, this field is only for printing
end

function ParallelDynamicalSystem(ds::CoreDynamicalSystem, states)
    f, st = parallel_rule(ds, states)
    if ds isa DeterministicIteratedMap
        pds = DeterministicIteratedMap(f, st, current_parameters(ds); t0 = initial_time(ds))
    elseif ds isa CoupledODEs
        T = eltype(first(st))
        prob = ODEProblem{true}(f, st, (T(initial_time(ds)), T(Inf)), current_parameters(ds))
        inorm = prob.u0 isa Matrix ? matrixnorm : vectornorm
        pds = CoupledODEs(prob, ds.diffeq; internalnorm = inorm)
    end
    M = ds isa CoupledODEs && isinplace(ds)
    return ParallelDynamicalSystemAnalytic{typeof(pds), M}(pds, dynamic_rule(ds))
end

# Out of place: everywhere the same
function parallel_rule(ds::CoreDynamicalSystem{false}, states)
    f = dynamic_rule(ds)
    S = typeof(correct_state(Val{false}(), first(states)))
    st = [S(s) for s in states]
    L = length(st)
    parallel_f = (du, u, p, t) -> begin
        @inbounds for i in 1:L
            du[i] = f(u[i], p, t)
        end
    end
    return parallel_f, st
end

# In place: for where `Vector{Vector}` is possible
function parallel_rule(ds::DeterministicIteratedMap{true}, states)
    f = dynamic_rule(ds)
    S = typeof(correct_state(Val{true}(), first(states)))
    st = [S(s) for s in states]
    L = length(st)
    parallel_f = (du, u, p, t) -> begin
        @inbounds for i in 1:L
            f(du[i], u[i], p, t)
        end
    end
    return parallel_f, st
end

# In place, uses matrix with each column a parallel state
function parallel_rule(ds::CoupledODEs{true}, states)
    st = Matrix(hcat(states...))
    f = dynamic_rule(ds)
    parallel_f = (du, u, p, t) -> begin
        for i in axes(st, 2)
            f(view(du, :, i), view(u, :, i), p, t)
        end
    end
    return parallel_f, st
end

###########################################################################################
# Analytically knwon rule: extensions
###########################################################################################
for f in (:(SciMLBase.step!), :current_time, :initial_time, :isdiscretetime, :reinit!,
        :current_parameters, :initial_parameters
    )
    @eval $(f)(pdsa::ParallelDynamicalSystemAnalytic, args...; kw...) = $(f)(pdsa.ds, args...; kw...)
end

(pdsa::ParallelDynamicalSystemAnalytic)(t::Real, i::Int = 1) = pdsa.ds(t)[i]
dynamic_rule(pdsa::ParallelDynamicalSystemAnalytic) = pdsa.original_f

# States IO for vector of vectors state
"""
    current_states(pds::ParallelDynamicalSystem)

Return an iterator over the parallel states of `pds`.
"""
current_states(pdsa::ParallelDynamicalSystemAnalytic) = current_state(pdsa.ds)

"""
    initial_states(pds::ParallelDynamicalSystem)

Return an iterator over the initial parallel states of `pds`.
"""
initial_states(pdsa::ParallelDynamicalSystemAnalytic) = initial_state(pdsa.ds)
function set_state!(pdsa::ParallelDynamicalSystemAnalytic, u, i::Int = 1)
    current_states(pdsa)[i] = u
    set_state!(pdsa.ds, current_states(pdsa))
end

# States IO for matrix state
# Unfortunately the `eachcol` generator cannot be accessed with `i`
# so we need to extend every method manually
# TODO: In Julia 1.9 `eachcol` can be indexed normally.
const PDSAM{D} = ParallelDynamicalSystemAnalytic{D, true}
current_states(pdsa::PDSAM) = eachcol(current_state(pdsa.ds))
current_state(pdsa::PDSAM, i::Int = 1) = view(current_state(pdsa.ds), :, i)
initial_states(pdsa::PDSAM) = eachcol(initial_state(pdsa.ds))
initial_state(pdsa::PDSAM, i::Int = 1) = view(initial_state(pdsa.ds), :, i)
(pdsa::PDSAM)(t::Real, i::Int = 1) = view(pdsa.ds(t), :, i)
function set_state!(pdsa::PDSAM, u, i::Int = 1)
    current_state(pdsa, i) .= u
    u_modified!(pdsa.ds.integ, true)
end

###########################################################################################
# Generic discrete time system: creation & extension
###########################################################################################
struct ParallelDiscreteTimeDynamicalSystem{D <: DynamicalSystem} <: ParallelDynamicalSystem
    systems::Vector{D}
end
const PDTDS = ParallelDiscreteTimeDynamicalSystem

function ParallelDynamicalSystem(ds::DiscreteTimeDynamicalSystem, states)
    systems = [deepcopy(ds) for s in states]
    for (i, s) in enumerate(states); reinit!(systems[i], s); end
    return ParallelDiscreteTimeDynamicalSystem(systems)
end

for f in (:current_time, :initial_time, :isdiscretetime,
        :current_parameters, :initial_parameters, :dynamic_rule,
    )
    @eval $(f)(pdtds::PDTDS, args...; kw...) = $(f)(pdtds.systems[1], args...; kw...)
end

(pdtds::PDTDS)(t::Real, i::Int = 1) = pdtds.systems[i](t)

function step!(pdtds::PDTDS, N::Int = 1, stop_at_dt::Bool = true)
    for ds in pdtds.systems
        step!(ds, N)
    end
    return
end

# Getting states also needs to be adjusted
for f in (:current_state, :initial_state)
    @eval $(f)(pdtds::PDTDS, i::Int = 1) = $(f)(pdtds.systems[i])
end
current_states(pdtds::PDTDS) = [current_state(ds) for ds in pdtds.systems]
initial_states(pdtds::PDTDS) = [initial_state(ds) for ds in pdtds.systems]

# Set stuff
set_parameter!(pdtds::PDTDS) = for ds in pdtds.systems; set_parameter!(ds, args...); end
set_state!(pdtds::PDTDS, u, i::Int = 1) = set_state!(pdtds.systems[i], u)
function SciMLBase.reinit!(pdtds::PDTDS, states = initial_states(pdtds); kwargs...)
    for (ds, s) in zip(pdtds.systems, states); reinit!(ds, s; kwargs...); end
end
