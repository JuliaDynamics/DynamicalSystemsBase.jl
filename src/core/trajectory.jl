export trajectory

"""
    trajectory(ds::DynamicalSystem, T [, u0]; kwargs...) → X, t

Evolve `ds` for a total time of `T` and return its trajectory `X`, sampled at equal time
intervals, and corresponding time vector.
Optionally provide a starting state `u0` which is `current_state(ds)` by default.

The returned time vector is `t = (t0+Ttr):Δt:(t0+Ttr+T)`.

If time evolution diverged before `T`, the remaining of the trajectory is set
to the last valid point.

`trajectory` is a very simple function provided for convenience.
For continuous time systems, it doesn't play well with callbacks,
use `DifferentialEquations.solve` if you want a trajectory/timeseries
that works with callbacks.

## Keyword arguments

* `Δt`:  Time step of value output. For discrete time systems it must be an integer.
  Defaults to `0.1` for continuous and `1` for discrete time systems. If you don't
  have access to unicode, the keyword `Dt` can be used instead.
* `Ttr = 0`: Transient time to evolve the initial state before starting saving states.
* `t0 = initial_time(ds)`: Starting time.
* `save_idxs::AbstractVector`: Which variables to output in `X`. It can be
  any type of index that can be given to [`observe_state`](@ref).
  Defaults to `1:dimension(ds)` (all dynamic variables).
  Note: if you mix integer and symbolic indexing be sure to initialize the array
  as `Any` so that integers `1, 2, ...` are not converted to symbolic expressions.
"""
function trajectory(ds::DynamicalSystem, T, u0 = initial_state(ds);
        save_idxs = nothing, t0 = initial_time(ds), kwargs...
    )
    accessor = svector_access(save_idxs)
    reinit!(ds, u0; t0)
    if isdiscretetime(ds)
        trajectory_discrete(ds, T; accessor, kwargs...)
    else
        trajectory_continuous(ds, T; accessor, kwargs...)
    end
end

function trajectory_discrete(ds, T;
        Dt::Integer = 1, Δt::Integer = Dt, Ttr::Integer = 0, accessor = nothing
    )
    ET = eltype(current_state(ds))
    t0 = current_time(ds)
    tvec = (t0+Ttr):Δt:(t0+T+Ttr)
    L = length(tvec)
    X = isnothing(accessor) ? dimension(ds) : length(accessor)
    data = Vector{SVector{X, ET}}(undef, L)
    Ttr ≠ 0 && step!(ds, Ttr)
    data[1] = obtain_state(current_state(ds), accessor)
    for i in 2:L
        step!(ds, Δt)
        data[i] = SVector{X, ET}(obtain_state(ds, current_time(ds), accessor))
        if !successful_step(ds)
            # Diverged trajectory; set final state to remaining set
            # and exit iteration early
            for j in (i+1):L
                data[j] = data[i]
            end
            break
        end
    end
    return StateSpaceSet(data), tvec
end

function trajectory_continuous(ds, T; Dt = 0.1, Δt = Dt, Ttr = 0.0, accessor = nothing)
    D = dimension(ds)
    t0 = current_time(ds)
    tvec = (t0+Ttr):Δt:(t0+T+Ttr)
    X = isnothing(accessor) ? D : length(accessor)
    ET = eltype(current_state(ds))
    sol = Vector{SVector{X, ET}}(undef, length(tvec))
    step!(ds, Ttr)
    for (i, t) in enumerate(tvec)
        while t > current_time(ds)
            step!(ds)
            successful_step(ds) || break
        end
        sol[i] = SVector{X, ET}(obtain_state(ds, t, accessor))
        if !successful_step(ds)
            # Diverged trajectory; set final state to remaining set
            # and exit iteration early
            for j in (i+1):length(tvec)
                sol[j] = sol[i]
            end
            break
        end

    end
    return StateSpaceSet(sol), tvec
end

# Util functions for `trajectory` to make indexing static vectors
# as efficient as possible
svector_access(::Nothing) = nothing
svector_access(x::AbstractVector{Int}) = SVector{length(x), Int}(x...)
svector_access(x::Int) = SVector{1, Int}(x)
svector_access(x::AbstractVector) = x

# this is a special wrapper around `observe_state` to make indexing
# static arrays more performant without unecessary allocations
obtain_state(u, ::Nothing) = u
obtain_state(u, i::SVector) = u[i]
function obtain_state(ds::DynamicalSystem, t::Real, i::Union{Nothing, SVector})
    return obtain_state(ds(t), i)
end
function obtain_state(ds::DynamicalSystem, t::Real, idxs::AbstractVector)
    u = ds(t)
    return [observe_state(ds, i, u, t) for i in idxs]
end
