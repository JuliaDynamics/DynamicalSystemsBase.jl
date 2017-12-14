__precompile__()

"""
Definitions of core system and data types used
in in the ecosystem of DynamicalSystems.jl
"""
module DynamicalSystemsDef

abstract type DynamicalSystem end

export DynamicalSystem, Systems

include("dataset.jl")
include("reconstruction.jl")
include("various.jl")

include("discrete.jl")
include("continuous.jl")
include("famous_systems.jl")

end
