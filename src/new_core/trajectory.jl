export trajectory

"""
    trajectory(ds::DynamicalSystem, T [, u]; kwargs...) → X, t

Return a dataset that will contain the trajectory of the system `ds`,
after evolving it for total time `T`. If `u` is given, `ds` is [`reinit!`](@ref) at `u`,
otherwise evolution starts from current status of `ds`.

See [`Dataset`](@ref) for info on how to use the returned object.

The returned time vector is `t = (t0+Ttr):Δt:(t0+Ttr+T)` and is not returned.
(`t0` is the [`current_time`](@ref) of `ds` which is by default `0`).

## Keyword arguments

* `Δt`:  Time step of value output. For discrete systems it must be an integer.
  Defaults to `0.01` for continuous and `1` for discrete time systems.
* `Ttr=0`: Transient time to evolve the initial state before starting saving states.
* `save_idxs::AbstractVector{Int}`: Which variables to output in the dataset (by default all).
"""
function trajectory(ds::DynamicalSystem, args...; save_idxs = nothing, kwargs...)
    accessor = svector_access(save_idxs)
    if isdiscretetime(ds)
        trajectory_discrete(ds, args...; accessor, kwargs...)
    else
        trajectory_continuous(ds, args...; accessor, kwargs...)
    end
end

function trajectory_discrete(ds, t, u0 = nothing;
        Δt::Integer = 1, Ttr::Integer = 0, accessor = nothing
    )
    !isnothing(u0) && reinit!(ds, u0)
    ET = eltype(current_state(ds))
    t0 = current_time(ds)
    tvec = (t0+Ttr):Δt:(t0+t+Ttr)
    L = length(tvec)
    T = eltype(current_state(ds))
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

function trajectory_continuous(integ, T, u0 = nothing;
        Δt = 0.01, Ttr = 0.0, accessor = nothing
    )
    !isnothing(u0) && reinit!(integ, u0)
    D = dimension(integ)
    t0 = current_time(integ)
    tvec = (t0+Ttr):Δt:(t0+T+Ttr)
    X = isnothing(accessor) ? D : length(accessor)
    ET = eltype(current_state(integ))
    sol = Vector{SVector{X, ET}}(undef, length(tvec))
    step!(integ, Ttr)
    for (i, t) in enumerate(tvec)
        while t > current_time(integ)
            step!(integ)
        end
        sol[i] = SVector{X, ET}(obtain_access(integ(t), accessor))
    end
    return Dataset(sol), tvec
end

# Util functions for `trajectory`
svector_access(::Nothing) = nothing
svector_access(x::AbstractArray) = SVector{length(x), Int}(x...)
svector_access(x::Int) = SVector{1, Int}(x)
obtain_access(u, ::Nothing) = u
obtain_access(u, i::SVector) = u[i]
