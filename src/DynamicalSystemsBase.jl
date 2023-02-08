module DynamicalSystemsBase

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end DynamicalSystemsBase

using Reexport
@reexport using StateSpaceSets
export Systems

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

include("predefined/famous_systems.jl")

include("deprecations.jl")

# include("core/dynamicalsystem.jl")
# include("core/create_dynamics.jl")
# include("core/api_docstrings.jl")
# include("core/discrete.jl")
# include("core/continuous.jl")
# include("core/pretty_printing.jl")

# include("advanced_integrators/projected_integrator.jl")
# include("advanced_integrators/stroboscopic_map.jl")
# include("advanced_integrators/poincare.jl")

# include("predefined/famous_systems.jl") # includes both discrete and continuous

# export GeneralizedDynamicalSystem, DynamicalSystem
# export ContinuousDynamicalSystem, DiscreteDynamicalSystem
# export rulestring, isdiscretetime
# export dimension, get_state, get_states
# export get_parameter, get_parameters, set_parameter!
# export trajectory, jacobian
# export integrator, tangent_integrator, parallel_integrator
# export set_state!, get_state, get_deviations, set_deviations!, current_time
# export SciMLBase, init, step!, isinplace, reinit!
# export Systems
# export projected_integrator, stroboscopicmap

end
