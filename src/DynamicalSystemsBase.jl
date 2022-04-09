"""
Definition of `DynamicalSystem` as well as all integrators
used in the ecosystem of DynamicalSystems.jl.

Also contains predefined well-known systems.
"""
module DynamicalSystemsBase

using DelayEmbeddings
import SciMLBase
# Import some core functions that are extended a lot:
import SciMLBase: init, step!, isinplace, reinit!, u_modified!

include("deprecations.jl")

include("core/dynamicalsystem.jl")
include("core/create_dynamics.jl")
include("core/api_docstrings.jl")
include("core/discrete.jl")
include("core/continuous.jl")
include("core/pretty_printing.jl")

include("advanced_integrators/projected_integrator.jl")
include("advanced_integrators/stroboscopic_map.jl")

include("predefined/famous_systems.jl") # includes both discrete and continuous

export GeneralizedDynamicalSystem, DynamicalSystem
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export get_rule_for_print, isdiscretetime
export dimension, get_state, get_states
export set_parameter!, trajectory, jacobian
export integrator, tangent_integrator, parallel_integrator
export set_state!, get_state, get_deviations, set_deviations!, current_time
export SciMLBase, init, step!, isinplace, reinit!
export Systems
export SVector, SMatrix, @SVector, @SMatrix
export projected_integrator, stroboscopicmap

end
