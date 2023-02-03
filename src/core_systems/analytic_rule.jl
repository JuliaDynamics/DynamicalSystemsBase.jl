export AnalyticRuleSystem

"""
    AnalyticRuleSystem

Abstract type meaning either [`DeterministicIteratedMap`](@ref) or [`CoupledODEs`](@ref).
"""
AnalyticRuleSystem{IIP} = Union{CoupledODEs{IIP}, DeterministicIteratedMap{IIP}}
