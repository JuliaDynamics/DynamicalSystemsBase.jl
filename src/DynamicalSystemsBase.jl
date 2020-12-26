"""
Definition of `DynamicalSystem` as well as all integrators
used in the ecosystem of DynamicalSystems.jl.

Also contains predefined well-known systems.
"""
module DynamicalSystemsBase

using DelayEmbeddings

include("core/dynamicalsystem.jl")
include("core/discrete.jl")
include("core/continuous.jl")
include("predefined/famous_systems.jl")

include("deprecations.jl")

export Systems, reinit!
export SVector, SMatrix, @SVector, @SMatrix

end
