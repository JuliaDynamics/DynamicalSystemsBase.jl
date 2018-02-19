ContinuousDS(args...; kwargs...) = (warn(
"ContinuousDS has changed to ContinuousDynamicalSystem"
); error("ContinuousDS syntax is not valid anymore.

Please see the documentation string of `DynamicalSystem`"))

DiscreteDS(args...; kwargs...) = (warn(
"ContinuousDS has changed to ContinuousDynamicalSystem"
); error("DiscreteDS syntax is not valid anymore.

Please see the documentation string of `DynamicalSystem`"))

evolve(args...; kwargs...) = error("Function `evolve` is no longer used.")
evolve!(args...; kwargs...) = error("Function `evolve!` is no longer used.")

export evolve, evolve!, ContinuousDS, DiscreteDS
