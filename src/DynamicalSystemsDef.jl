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
