module DynamicalSystemsBase

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DynamicalSystemsBase

using Reexport
@reexport using StateSpaceSets

import SciMLBase
import SciMLBase: reinit!, step!, isinplace
using SciMLBase: recursivecopy
export reinit!, step!, isinplace, recursivecopy

include("core/dynamicalsystem_interface.jl")
include("core/trajectory.jl")
include("core/utilities.jl")
include("core/pretty_printing.jl")

include("core_systems/discrete_time_map.jl")
include("core_systems/continuous_time_ode.jl")
include("core_systems/arbitrary_steppable.jl")
include("core_systems/additional_supertypes.jl")

include("derived_systems/stroboscopic_map.jl")
include("derived_systems/parallel_systems.jl")
include("derived_systems/tangent_space.jl")
include("derived_systems/poincare/poincaremap.jl")
include("derived_systems/projected_system.jl")

include("deprecations.jl")

end
