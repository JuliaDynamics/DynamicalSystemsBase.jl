# Parallel integration is a bit special;
# For `AnalyticRuleSystem` a dedicated structure exists that uses the existing
# integrators with a vector of vectors.
# For all discrete time systems another structure exists that deepcopies the systems.
# And for all continuous time systems another structure exists.
# TODO: Continous time utilizing `step!(integ, dt, true)` and requiring a `dt`.
export ParallelDynamicalSystem, current_states, initial_states

"""
    ParallelDynamicalSystem(ds::DynamicalSystem, states::AbstractVector)

A struct that evolves several `states` of a given dynamical system in parallel
**at exactly the same times**. Useful when wanting to evolve several different trajectories
of the same system while ensuring that they share parameters and time vector.

This struct follows the [`DynamicalSystem`](@ref) interface with the following adjustments:

- The function [`current_state`](@ref) is called as `current_state(pe, i::Int = 1)`
  which returns the `i`th state. Same for [`initial_state`](@ref).
- Similarly, [`set_state`](@ref) obtains a second argument `i::Int = 1` to
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

##################################################################################
# Analytically knwon rule
##################################################################################
# We don't parameterize the dimension because it does not need to be known
# at compile time given the usage of the integrator.
# It uses the generic `DynamicalSystem` dispatch
struct ParallelDynamicalSystemAnalytic{D} <: ParallelDynamicalSystem
    ds::D
    original_f # no type parameterization here, this field is only for printing
end

function ParallelDynamicalSystem(ds::AnalyticRuleSystem, states)
    f, st = isinplace(ds) ? parallel_f_iip(ds, states) : parallel_f_oop(ds, states)
    if ds isa DeterministicIteratedMap
        pds = DeterministicIteratedMap(f, st, current_parameters(ds), initial_time(ds))
    elseif ds isa CoupledODEs
        pds = CoupledODEs(
            f, st, current_parameters(ds); t0 = initial_time(ds), diffeq = ds.diffeq
        )
    end
    return ParallelDynamicalSystemAnalytic(pds, dynamic_rule(ds))
end

# Extensions
dynamic_rule(pdsa::ParallelDynamicalSystemAnalytic) = pdsa.original_f
current_states(pdsa::ParallelDynamicalSystemAnalytic) = current_state(pdsa.ds)
initial_states(pdsa::ParallelDynamicalSystemAnalytic) = initial_state(pdsa.ds)
function set_state!(pdsa::ParallelDynamicalSystemAnalytic, u, i::Int = 1)
    current_states(pdsa)[i] = u
    set_state!(pdsa.ds, current_states(pdsa))
end

for f in (:(SciMLBase.step!), :current_time, :initial_time, :isdiscretetime, :reinit!,
        :current_parameters, :initial_parameters
    )
    @eval $(f)(pdsa::ParallelDynamicalSystemAnalytic, args...; kw...) = $(f)(pdsa.ds, args...; kw...)
end

(pdsa::ParallelDynamicalSystemAnalytic)(t::Real, i::Int = 1) = pdsa.ds(t)[i]

function parallel_f_iip(ds, states)
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

function parallel_f_oop(ds, states)
    S = typeof(correct_state(Val{false}(), first(states)))
    st = [S(s) for s in states]
    L = length(st)
    parallel_f = (du, u, p, t) -> begin
        @inbounds for i in 1:L
            du[i] = ds.f(u[i], p, t)
        end
    end
    return parallel_f, st
end


# struct ParallelItegratorDiscrete
# #     systems
# # end

