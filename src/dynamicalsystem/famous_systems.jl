"""
Sub-module of the module `DynamicalSystemsBase`, which contains pre-defined
famous systems.
"""
module Systems
using DynamicalSystemsBase

using StaticArrays
const twopi = 2π

using DynamicalSystemsBase: DDS
include("discrete_famous_systems.jl")

using DynamicalSystemsBase: CDS
include("continuous_famous_systems.jl")

end# Systems module
