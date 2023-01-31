# Parallel integration is a bit special;
# For `CoupledODEs` and `DeterministicIteratedMap` a dedicated structure exists;
# For all discrete time systems another structure exists.
# And for all continuous time systems another structure exists.
# TODO: Continous time utilizing `step!(integ, dt, true)` and requiring a `dt`.

"""
    ParallelEvolver(ds::DynamicalSystem, states)

A struct that evolves several `states` of a given dynamical system in parallel
**at exactly the same times**. Useful when wanting to evolve several different trajectories
of the same system while ensuring that they share parameters and time vector.

This struct follows the [`DynamicalSystem`](@ref) interface with the following adjustments:

- The function [`current_state`](@ref) is called as `current_state(pe, i::Int = 1)`
  which returns the `i`th state. Same for [`initial_state`](@ref).
- Similarly, [`set_state`](@ref) obtains a second argument `i::Int = 1` to
  set the `i`-th state.
"""
abstract type ParallelEvolver <: DynamicalSystem end

# Generic interface that doesn't depend on implementation
isinplace(::ParallelEvolver) = true
current_state(p::ParalellEvolver, i::Int = 1) = current_states(p)[i]
initial_state(p::ParalellEvolver, i::Int = 1) = initial_states(p)[i]

##################################################################################
# Analytically knwon rule
##################################################################################
# We don't parameterize the dimension because it does not need to be known
# at compile time given the usage of the integrator. Besides, the field `ds`
# has it as type parameter...

struct ParallelEvolverAnalytic{D, U} <: ParallelEvolver
    ds::D
end

function ParallelEvolver(ds::AnalyticRuleSystem, states)
    f, st = isinplace(ds) ? parallel_f_iip(ds, states) : parallel_f_oop(ds, states)
    if ds isa DeterministicIteratedMap
        pds = DeterministicIteratedMap(f, st, current_parameters(ds), initial_time(ds))
    elseif ds isa CoupledODEs
        pds = CoupledODEs(
            f, st, current_parameters(ds); t0 = initial_time(ds), diffeq = ds.diffeq
        )
    end
    return ParallelEvolverAnalytic(pds)
end

current_states(p::ParallelEvolverAnalytic) = current_state(p.ds)
initial_states(p::ParallelEvolverAnalytic) = initial_state(p.ds)
function set_state!(p::::ParallelEvolverAnalytic, i::Int = 1)
    fafafa
end




function parallel_f_iip(ds, states)
    f = dynamic_rule(ds)
    S = correct_state_type(Val{true}(), first(states))
    st = [S(s) for s in states]
    L = length(st)
    parallel_f = (du, u, p, t) -> begin
        @inbounds for i in 1:L
            f(du[i], u[i], p, t)
        end
    end
    return parallel_f, st
end

function parallel_f_oop(ds::DS{false}, states)
    S = correct_state_type(Val{false}(), first(states))
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

