export trajectory

"""
    trajectory(ds::DynamicalSystem, T [, u0]; kwargs...) → X, t

Return a dataset `X` that will contain the trajectory of the system `ds`,
after evolving it for total time `T`.
`u0` is the state given given to [`reinit!`](@ref) prior to time evolution
and defaults to [`initial_state(ds)`](@ref).

See [`Dataset`](@ref) for info on how to use `X`.
The returned time vector is `t = (t0+Ttr):Δt:(t0+Ttr+T)`.

## Keyword arguments

* `Δt`:  Time step of value output. For discrete time systems it must be an integer.
  Defaults to `0.1` for continuous and `1` for discrete time systems.
* `Ttr = 0`: Transient time to evolve the initial state before starting saving states.
* `t0 = initial_time(ds)`: Starting time.
* `save_idxs::AbstractVector{Int}`: Which variables to output in `X` (by default all).
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

function trajectory_discrete(ds, T; Δt::Integer = 1, Ttr::Integer = 0, accessor = nothing)
    ET = eltype(current_state(ds))
    t0 = current_time(ds)
    tvec = (t0+Ttr):Δt:(t0+T+Ttr)
    L = length(tvec)
    X = isnothing(accessor) ? dimension(ds) : length(accessor)
    data = Vector{SVector{X, ET}}(undef, L)
    Ttr ≠ 0 && step!(ds, Ttr)
    data[1] = obtain_access(current_state(ds), accessor)
    for i in 2:L
        step!(ds, Δt)
        data[i] = SVector{X, ET}(obtain_access(current_state(ds), accessor))
    end
    return Dataset(data), tvec
end

function trajectory_continuous(ds, T; Δt = 0.1, Ttr = 0.0, accessor = nothing)
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
        end
        sol[i] = SVector{X, ET}(obtain_access(ds(t), accessor))
    end
    return Dataset(sol), tvec
end

# Util functions for `trajectory`
svector_access(::Nothing) = nothing
svector_access(x::AbstractArray) = SVector{length(x), Int}(x...)
svector_access(x::Int) = SVector{1, Int}(x)
obtain_access(u, ::Nothing) = u
obtain_access(u, i::SVector) = u[i]
