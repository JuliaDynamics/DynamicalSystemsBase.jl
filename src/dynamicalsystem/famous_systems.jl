"""
Sub-module of the module `DynamicalSystemsBase`, which contains pre-defined
famous systems.
"""
module Systems
using DynamicalSystemsBase
using DynamicalSystemsBase: DDS
# using DynamicalSystemsBase: CDS

using StaticArrays
const twopi = 2π

include("discrete_famous_systems.jl")
# include("continuous_famous_systems.jl")

end# Systems module
