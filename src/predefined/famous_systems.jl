"""
Sub-module of the module `DynamicalSystemsBase`, which contains pre-defined
famous systems. This is provided purely as a convenience.
Nothing here is tested, nor guaranteed to be stable in future versions.
"""
module Systems
using DynamicalSystemsBase

using DynamicalSystemsBase: DDS
include("discrete_famous_systems.jl")

using DynamicalSystemsBase: CDS
include("continuous_famous_systems.jl")

end# Systems module
