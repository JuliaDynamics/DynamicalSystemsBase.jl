"""
Definition of `DynamicalSystem` as well as all integrators
used in the ecosystem of DynamicalSystems.jl.

Also contains predefined well-known systems.
"""
module DynamicalSystemsBase

using Reexport
@reexport using DelayEmbeddings

include("dynamicalsystem.jl")
include("discrete.jl")
include("continuous.jl")
include("famous_systems.jl")

include("deprecations.jl")

export Systems, reinit!
export SVector, SMatrix, @SVector, @SMatrix

end
