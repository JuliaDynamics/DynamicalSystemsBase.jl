using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, step!
import Base: eltype

export dimension, state, DynamicalSystem, integrator
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export set_parameter!, step!, jacobian, inittime

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

safe_state_type(ds::DS{true}, u0) = Vector(u0)
safe_state_type(ds::DS{false}, u0) = SVector{length(u0)}(u0...)
safe_state_type(ds::DS{false}, u0::Number) = u0

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################
"""
    DynamicalSystem

The central structure of **DynamicalSystems.jl**. All functions of the suite that
handle systems "analytically" (in the sense that they can use known equations of
motion) expect an instance of this type.

Contains a "problem" defining the system as well as the jacobian function.

## Constructing a `DynamicalSystem`
We suggest to use the constructors:
```julia
DiscreteDynamicalSystem(eom, state, p [, jacobian]; t0::Int = 0, J0)
ContinuousDynamicalSystem(eom, state, p [, jacobian]; t0 = 0.0, J0)
```
with `eom` the equations of motion function.
`p` is a parameter container, which we highly suggest to use a mutable object like
`Array`, [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl) or
a dictionary. Pass `nothing` in the place of `p` if your system does not have
parameters.

The first case creates an internal implementation of a discrete system,
which is as fast as possible. With these constructors you also do not need to
provide some final time, since it is not used by **DynamicalSystems.jl** in any manner.

### Equations of motion
The are two "versions" for `DynamicalSystem`, depending on whether the
equations of motion (`eom`) are in-place (iip) or out-of-place (oop).
Here is how to define them:

* **iip** : The `eom` **must** be in the form `eom(x, p, t) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an [`SVector`](http://juliaarrays.github.io/StaticArrays.jl/stable/pages/api.html#SVector-1)
  containing the next state.
* **oop** : The `eom` **must** be in the form `eom!(xnew, x, p, t)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

`t` stands for time (integer for discrete systems).
iip is suggested for big systems, whereas oop is suggested for small systems
(break-even point at around 10 dimensions).

The constructor deduces automatically whether the EOM are iip or oop. It is not
possible however to deduce whether the system is continuous or discrete just from the
equations of motion, hence the 2 constructors.

### Jacobian
The final optional argument (`jacobian`) for the constructors
is a *function* and (if given) must also be of the same form as the `eom`,
`jacobian(x, p, n) -> [SMatrix](http://juliaarrays.github.io/StaticArrays.jl/stable/pages/api.html#SMatrix-1)`
for the out-of-place version and `jacobian!(xnew, x, p, n)` for the in-place version.

If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).

`J0` is a keyword argument allowing you to pass a pre-initialized Jacobian matrix
(very helpful for large systems).

### Using `DEProblem`
You can always create a `DynamicalSystem` with the constructor
```julia
DynamicalSystem(prob::DEProblem [, jacobian]; J0)
```
if you have
an instance of `DEProblem`, because you may want to use the callback functionality of
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/). Notice however,
that most functions that use known equations of motion (and thus use `DynamicalSystem`)
expect the vector field to be differentiable,
limiting the use of callbacks significantly.

We give no guarantees that *any* function of **DynamicalSystems.jl** suite will
give "reasonable" results when using callbacks of any kind.

## Relevant Functions
[`jacobian`](@ref), [`state`](@ref), [`trajectory`](@ref),
[`set_parameter!`](@ref), [`integrator`](@ref), [`tangent_integrator`](@ref),
[`parallel_integrator`](@ref).
"""
struct DynamicalSystem{
        IIP, # is in place , for dispatch purposes and clarity
        IAD, # is auto differentiated? Only for constructing tangent_integrator
        PT<:DEProblem, # problem type
        JAC, # jacobian function (either user-provided or FD)
        JM}  # initialized jacobian matrix
    prob::PT
    jacobian::JAC
    J::JM
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
    return DynamicalSystem(prob, jacobian)
end
function DynamicalSystem(prob::DEProblem, jacobian::JAC;
    J0 = nothing) where {JAC}

    if J0 == nothing
        J = get_J(prob, jacobian)
    else
        J = J0
    end
    IIP = isinplace(prob)
    JIP = isinplace(jacobian, 4)
    JIP == IIP || throw(ArgumentError(
    "The jacobian function and the equations of motion are not of the same form!"*
    " The jacobian `isinlace` was $(JIP) while the eom `isinplace` was $(IIP)."))
    DEP = typeof(prob)
    JM = typeof(J)
    return DynamicalSystem{IIP, false, DEP, JAC, JM}(prob, jacobian, J)
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
function create_jacobian(prob) #creates jacobian function
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
# Gets the jacobian at current state
function get_J(prob, jacob)
    D = dimension(prob)
    if isinplace(prob)
        J = similar(prob.u0, (D,D))
        jacob(J, prob.u0, prob.p)
    else
        J = jacob(prob.u0, prob.p)
    end
    return J
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
