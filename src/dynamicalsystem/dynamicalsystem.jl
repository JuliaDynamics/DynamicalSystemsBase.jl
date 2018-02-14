using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, step!
import Base: eltype

export dimension, state, DynamicalSystem, integrator
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export set_parameter!, step!, jacobian, inittime

#######################################################################################
#                          Basic functions and interface                              #
#######################################################################################
dimension(prob::DEProblem) = length(prob.u0)
eltype(prob::DEProblem) = eltype(prob.u0)
state(prob::DEProblem) = prob.u0
state(integ::DEIntegrator) = integ.u
hascallback(prob::DEProblem) = :callback ∈ fieldnames(prob) && prob.callback != nothing
statetype(prob::DEProblem) = eltype(prob.u0)
systemtype(::ODEProblem) = "continuous"
systemtype(::DiscreteProblem) = "discrete"
inittime(prob::DEProblem) = prob.tspan[1]

"""
    step!(integrator [, Δt])
Step forwards the `integrator` once. If `Δt` is given, step until there is a
time difference ≥ Δt from before starting stepping.
"""
function step!(integ, Δt::Real)
    t = integ.t
    while integ.t < t + Δt
        step!(integ)
    end
end

safe_state_type(IIP, u0) = IIP ? Vector(u0) : SVector{length(u0)}(u0...)

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################
"""
    DynamicalSystem(prob::DEProblem [, jacobian])

The central structure of **DynamicalSystems.jl**. All functions of the suite that
handle systems "analytically" (in the sense that they can use known equations of
motion) expect an instance of this type.

Contains a "problem" defining the system as well as the jacobian function.

## Description
You can always create a `DynamicalSystem` with the above constructor if you have
an instance of `DEProblem`, because you may want to use the callback functionality of
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/). Notice however,
that most functions that use known equations of motion (and thus use `DynamicalSystem`)
expect the vector field to be differentiable,
limiting the use of callbacks significantly.

We suggest to use the constructors:
```julia
DiscreteDynamicalSystem(eom, state, p [, jacobian]; t0::Int = 0)
ContinuousDynamicalSystem(eom, state, p [, jacobian]; t0 = 0.0)
```
The first case creates an internal implementation of a discrete system,
which is as fast as possible. With these constructors you also do not need to
provide some final time, since it is not used by **DynamicalSystems.jl** in any manner.

The are two "versions" for `DynamicalSystem`, depending on whether the
equations of motion (`eom`) are in-place (iip) or out-of-place (oop).
Here is how to define them:

* **iip** : The `eom` **must** be in the form `eom(x, p) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector` containing the next state.
* **oop** : The `eom` **must** be in the form `eom!(xnew, x, p)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

iip is suggested for big systems, whereas oop is suggested for small systems.

The constructor deduces automatically whether the EOM are iip or oop. It is not
possible however to deduce whether the system is continuous or discrete just from the
equations of motion, hence the 2 constructors.

## Relevant Functions
[`jacobian`](@ref), [`state`](@ref), [`trajectory`](@ref),
[`set_parameter!`](@ref), [`integrator`](@ref), [`tangent_integrator`](@ref),
[`parallel_integrator`](@ref).
"""
struct DynamicalSystem{
        IIP, # is in place , for dispatch purposes and clarity
        IAD, # is auto differentiated? Only for constructing tangent_integrator
        PT<:DEProblem, # problem type
        JAC} # jacobian function (either user-provided or FD)
    prob::PT
    jacobian::JAC
end

DS = DynamicalSystem
isautodiff(ds::DS{IIP, IAD, DEP, JAC}) where {DEP, IIP, JAC, IAD} = IAD
problemtype(ds::DS{IIP, IAD, DEP, JAC}) where {DEP<:DiscreteProblem, IIP, JAC, IAD} =
DiscreteProblem
problemtype(ds::DS{IIP, IAD, DEP, JAC}) where {DEP<:ODEProblem, IIP, JAC, IAD} =
ODEProblem

function DynamicalSystem(prob::DEProblem)
    IIP = isinplace(prob)
    jacobian = create_jacobian(prob)
    DEP = typeof(prob)
    JAC = typeof(jacobian)
    return DynamicalSystem{IIP, true, DEP, JAC}(prob, jacobian)
end
function DynamicalSystem(prob::DEProblem, jacobian::JAC) where {JAC}
    IIP = isinplace(prob)
    JIP = isinplace(jacobian, 4)
    JIP == IIP || throw(ArgumentError(
    "The jacobian function and the equations of motion are not of the same form!"*
    " The jacobian `isinlace` was $(JIP) while the eom `isinplace` was $(IIP)."))
    DEP = typeof(prob)
    return DynamicalSystem{IIP, false, DEP, JAC}(prob, jacobian)
end

# Expand methods
for f in (:isinplace, :dimension, :eltype, :statetype, :state, :systemtype,
    :set_parameter!, :inittime)
    @eval begin
        @inline ($f)(ds::DynamicalSystem, args...) = $(f)(ds.prob, args...)
    end
end


#####################################################################################
#                                    Jacobians                                      #
#####################################################################################
function create_jacobian(prob)
    IIP = isinplace(prob)
    if IIP
        dum = deepcopy(prob.u0)
        cfg = ForwardDiff.JacobianConfig(
            (y, x) -> prob.f(y, x, prob.p, prob.tspan[1]),
            dum, prob.u0)
        jacobian = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> prob.f(y, x, p, t),
        dum, u, cfg, Val{false}())
    else
        # SVector methods do *not* use the config
        # cfg = ForwardDiff.JacobianConfig(
        #     (x) -> prob.f(x, prob.p, prob.tspan[1]), prob.u0)
        jacobian = (u, p, t) ->
        ForwardDiff.jacobian((x) -> prob.f(x, p, t), u, #=cfg=#)
    end
    return jacobian
end

# Jacobian function application
"""
    jacobian(ds::DynamicalSystem, u = ds.prob.u0)
Return the jacobian of the system at `u` (at initial time).
"""
function jacobian(ds::DS{true}, u = ds.prob.u0)
    D = dimension(ds)
    J = similar(u, D, D)
    ds.jacobian(J, u, ds.prob.p, inittime(ds.prob))
    return J
end
jacobian(ds::DS{false}, u = ds.prob.u0) =
ds.jacobian(u, ds.prob.p, inittime(ds.prob))

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds)*" dynamical system"

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(ds.jacobian)"

function Base.show(io::IO, ds::DS)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    rpad(" jacobian: ", ps)*"$(jacobianstring(ds))\n"
    )
end

#######################################################################################
#                                     Docstrings                                      #
#######################################################################################
"""
    integrator(ds::DynamicalSystem; diff_eq_kwargs) -> integ
Return a `DEIntegrator` object that can be used to evolve a system interactively
using `step!(integ [, Δt])`.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function integrator end

"""
    tangent_integrator(ds::DynamicalSystem, Q0 | k::Int; u0, diff_eq_kwargs)
Return a `DEIntegrator` object that evolves in parallel both the system as well
as deviation vectors living on the tangent space.

`Q0` is a *matrix* whose columns are initial values for deviation vectors `ws`. If
instead of a matrix `Q0` an integer `k` is given, then `k` random orthonormal
vectors are choosen as initial conditions. You can also give as a keyword argument
a different initial state for the system `u0`.

See [`trajectory`](@ref) for `diff_eq_kwargs`.

## Description

If ``J`` is the jacobian of the system then the equations for the system
and a deviation vector (or matrix) ``w`` are:
```math
\\begin{aligned}
u_{n+1} &= f(u_n, p, t) \\\\
w_{n+1} &= J(u_n, p, t) \\times w_n
\\end{aligned}
```
with ``f`` being the equations of motion and ``u`` the system state.

*Note* - the example shown is for discrete systems, for continuous use ``du/dt``.
"""
function tangent_integrator end

#####################################################################################
#                                    Auxilary                                       #
#####################################################################################
"""
    set_parameter!(ds::DynamicalSystem, index, value)
    set_parameter!(ds::DynamicalSystem, values)
Change one or many parameters of the system
by setting `p[index] = value` in the first case
and `p .= values` in the second.
"""
set_parameter!(prob, index, value) = (prob.p[index] = value)
set_parameter!(prob, values) = (prob.p .= values)
set_parameter!(ds::DS, args...) = set_parameter!(ds.prob, args...)
