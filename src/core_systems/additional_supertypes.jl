export CoreDynamicalSystem

"""
    CoreDynamicalSystem

Union type meaning either [`DeterministicIteratedMap`](@ref) or [`CoupledODEs`](@ref),
which are the core systems whose dynamic rule `f` is known analytically.

This type is used for deciding whether a creation of a [`TangentDynamicalSystem`](@ref)
is possible or not.
"""
CoreDynamicalSystem{IIP} = Union{CoupledODEs{IIP}, DeterministicIteratedMap{IIP}}
