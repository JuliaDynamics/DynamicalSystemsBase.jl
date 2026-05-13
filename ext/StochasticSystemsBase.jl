module StochasticSystemsBase

using DynamicalSystemsBase: DynamicalSystemsBase, SciMLBase, correct_state, CoupledODEs,
    CoupledSDEs, StateSpaceSets, isinplace, _delete, set_parameter!,
    set_parameters!, set_state!, dynamic_rule, isdeterministic, current_state,
    DynamicalSystemsBase, _set_parameter!, derivative_discontinuity!,
    additional_details, referrenced_sciml_prob, DEFAULT_DIFFEQ,
    SVector, SMatrix, current_parameters, initial_state, initial_time
using SciMLBase: SDEProblem, AbstractSDEIntegrator, __init, SDEFunction, step!
using StochasticDiffEq: SOSRA, setup_next_step!
using LinearAlgebra
import Random

include("src/CoupledSDEs.jl")
include("src/classification.jl")

export CoupledSDEs, CoupledODEs

end
