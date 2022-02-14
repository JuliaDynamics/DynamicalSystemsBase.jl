"""
Definition of `DynamicalSystem` as well as all integrators
used in the ecosystem of DynamicalSystems.jl.

Also contains predefined well-known systems.
"""
module DynamicalSystemsBase

using DelayEmbeddings

include("deprecations.jl")

include("core/dynamicalsystem.jl")
include("core/create_dynamics.jl")
include("core/api_docstrings.jl")
include("core/discrete.jl")
include("core/continuous.jl")
include("core/integrator_wrappers.jl")

include("predefined/famous_systems.jl") # includes both discrete and continuous

export Systems, reinit!
export SVector, SMatrix, @SVector, @SMatrix

end
