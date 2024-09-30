module StochasticSystemsBase

using DynamicalSystemsBase: DynamicalSystemsBase, SciMLBase, correct_state, CoupledODEs,
                            CoupledSDEs, StateSpaceSets, isinplace, _delete, set_parameter!,
                            set_state!, dynamic_rule, isdeterministic, current_state,
                            DynamicalSystemsBase, _set_parameter!, u_modified!,
                            additional_details, referrenced_sciml_prob, DEFAULT_DIFFEQ,
                            SVector, SMatrix, current_parameters
using SciMLBase: SDEProblem, AbstractSDEIntegrator, __init, SDEFunction, step!
using StochasticDiffEq: SOSRA
using LinearAlgebra

include("src/CoupledSDEs.jl")
include("src/classification.jl")

export CoupledSDEs, CoupledODEs

end
