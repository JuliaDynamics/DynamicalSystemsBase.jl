@deprecate get_state current_state
@deprecate get_deviations current_deviations

export integrator, tangent_integrator, parallel_integrator, poincaremap
export projected_integrator

function integrator(ds, args...; kwargs...)
    @warn "`integrator` is deprecated. Dynamical systems themselves are now integrators."
    return ds
end

for F in (:DiscreteDynamicalSystem, :ContinuousDynamicalSystem)
    @eval function $(F)(f, u0, p, J::Function)
        throw(ArgumentError("""
        Things have changed in DynamicalSystems.jl and now you cannot provide
        a Jacobian function as a 4th argument to $($(F)). You have to
        first initialize $($(F)) without a Jacobian, and then pass the initialized
        system and Jacobian into a `TangentDynamicalSystem`.
        """))
    end
end

function tangent_integrator(ds::DynamicalSystem, k; kwargs...)
    @warn("""
    `tangent_integrator` is no longer a valid name. We will create a
    `TangentDynamicalSystem` for you but keep in mind that this will get the automatic
    Jacobian via ForwardDiff.jl. If you have a hand-coded Jacobian,
    You need to provide it to the `TangentDynamicalSystem` constructor from now on.
    """
    )
    if k isa Int
        return TangentDynamicalSystem(ds; k = k)
    elseif k isa AbstractArray
        return TangentDynamicalSystem(ds; Q0 = k)
    else
        error("second argument isn't integer or array.")
    end
end

function parallel_integrator(ds::DynamicalSystem, states; kwargs...)
    @warn("""
    `parallel_integrator` is deprecated in favor of `ParallelDynamicalSystem`.
    It also doesn't accept keywords anymore.
    """
    )
    return ParalleDynamicalSystem(ds, states)
end

function projected_integrator(ds::DynamicalSystem, projection, complete_state; kwargs...)
    @warn("""
    `projected_integrator` is deprecated in favor of `ProjectedDynamicalSystem`.
    It also doesn't accept keywords anymore.
    """
    )
    return ParalleDynamicalSystem(ds, projection, complete_state)
end

function stroboscopicmap(ds::DynamicalSystem, T; kwargs...)
    @warn("""
    `stroboscopicmap` is deprecated in favor of `StroboscopicMap`.
    It also doesn't accept keywords anymore.
    """
    )
    return StroboscopicMap(ds, T)
end

function poincaremap(ds::DynamicalSystem, plane, Tmax=1e3; kwargs...)
    @warn("""
    `poincaremap` is deprecated in favor of `PoincareMap`.
    """
    )
    return PoincareMap(ds, plane; Tmax, kwargs...)
end