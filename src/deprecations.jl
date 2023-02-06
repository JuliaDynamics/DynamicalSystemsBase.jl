for F in (:DiscreteDynamicalSystem, :ContinuousDynamicalSystem)
    @eval function $(F)(f, u0, p, J::Function)
        throw(ArgumentError("""
        Things have changed in DynamicalSystems.jl and now you cannot provide
        a Jacobian function as a 4th argument to $(F). You have to
        first initialize $(F) without a Jacobian, and then pass the initialized
        system and Jacobian into a `TangentDynamicalSystem`.
        """))
    end
end