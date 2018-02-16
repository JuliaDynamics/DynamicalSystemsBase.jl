using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, step!
import Base: eltype

export dimension, state, DynamicalSystem, jacobian
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export set_parameter!, step!, inittime
export trajectory
export integrator, tangent_integrator, parallel_integrator

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################
"""
    DynamicalSystem

The central structure of **DynamicalSystems.jl**. All functions of the suite that
handle systems "analytically" (in the sense that they can use known equations of
motion) expect an instance of this type.

Contains a problem defining the system (field `prob`) and the jacobian function
(field `jacobian`).

## Constructing a `DynamicalSystem`
```julia
DiscreteDynamicalSystem(eom, state, p [, jacobian]; t0::Int = 0, J0)
ContinuousDynamicalSystem(eom, state, p [, jacobian]; t0 = 0.0, J0)
```
with `eom` the equations of motion function.
`p` is a parameter container, which we highly suggest to use a mutable object like
`Array`, [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl) or
a dictionary. Pass `nothing` in the place of `p` if your system does not have
parameters. With these constructors you also do not need to
provide some final time, since it is not used by **DynamicalSystems.jl** in any manner.

The keyword arguments `t0`, `J0` allow you to choose the initial time and provide
an initialized Jacobian matrix.

The discrete constructor creates an internal implementation of a discrete system,
which is as fast as possible.
The continuous uses [**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/),
see [`trajectory`](@ref) for default arguments.

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

### Using `DEProblem`
You can always create a `DynamicalSystem` with the constructor
```julia
DynamicalSystem(prob::DEProblem [, jacobian]; J0)
```
if you have
an instance of `DEProblem` (either discrete or continuous),
because you may want to use the callback functionality of
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/). Notice however
that callbacks do not propagate into methods that evolve tangent space,
like e.g. [`lyapunovs`](@ref) or [`gali`](@ref).

Using callbacks with **DynamicalSystems.jl** is very under-tested.
Use at your own risk!

### Getting a `Solution` struct
Provided that you do not use the internal fast implementation of
a discrete problem, you can *always* take advantage of the full capabilities of
the `Solution` struct of
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/). Just do:
```julia
typeof(ds) <: DynamicalSystem # true
sol = solve(ds.prob, alg; kwargs...)
# do stuff with sol
```

## Relevant Functions
[`state`](@ref), [`trajectory`](@ref), [`jacobian`](@ref), [`dimension`](@ref),
[`set_parameter!`](@ref).
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
isautodiff(ds::DS{IIP, IAD, DEP, JAC, JM}) where {DEP, IIP, JAC, IAD, JM} = IAD

function DynamicalSystem(prob::DEProblem)
    IIP = isinplace(prob)
    jac = create_jacobian(prob)
    DEP = typeof(prob)
    JAC = typeof(jac)
    return DynamicalSystem(prob, jac; iad = true)
end
function DynamicalSystem(prob::DEProblem, jac::JAC;
    J0 = nothing, iad = false) where {JAC}

    IIP = isinplace(prob)
    JIP = isinplace(jac, 4)
    JIP == IIP || throw(ArgumentError(
    "The jacobian function and the equations of motion are not of the same form!"*
    " The jacobian `isinplace` is $(JIP) while the eom `isinplace` is $(IIP)."))

    if J0 == nothing
        J = get_J(prob, jac)
        IIP || typeof(J) <: SMatrix || throw(ArgumentError(
        "The jacobian function must return an SMatrix for the out-of-place version"
        ))
    else
        J = safe_matrix_type(IIP, J0)
    end

    DEP = typeof(prob)
    JM = typeof(J)

    return DynamicalSystem{IIP, iad, DEP, JAC, JM}(prob, jac, J)
end

# Expand methods
for f in (:isinplace, :dimension, :statetype, :state, :systemtype,
    :set_parameter!, :inittime)
    @eval begin
        @inline ($f)(ds::DynamicalSystem, args...) = $(f)(ds.prob, args...)
    end
end

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
#                                    Jacobians                                        #
#######################################################################################
function create_jacobian(prob) #creates jacobian function
    IIP = isinplace(prob)
    if IIP
        dum = deepcopy(prob.u0)
        cfg = ForwardDiff.JacobianConfig(
            (y, x) -> prob.f(y, x, prob.p, prob.tspan[1]),
            dum, prob.u0)
        jac = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> prob.f(y, x, p, t),
        dum, u, cfg, Val{false}())
    else
        # SVector methods do *not* use the config
        # cfg = ForwardDiff.JacobianConfig(
        #     (x) -> prob.f(x, prob.p, prob.tspan[1]), prob.u0)
        jac = (u, p, t) ->
        ForwardDiff.jacobian((x) -> prob.f(x, p, t), u, #=cfg=#)
    end
    return jac
end
# Gets the jacobian at current state
function get_J(prob, jacob)
    D = dimension(prob)
    if isinplace(prob)
        J = similar(prob.u0, (D,D))
        jacob(J, prob.u0, prob.p, inittime(prob))
    else
        J = jacob(prob.u0, prob.p, inittime(prob))
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

#######################################################################################
#                            Tanget & Parallel Dynamics                               #
#######################################################################################
# Create equations of motion of tangent dynamics
function create_tangent(ds::DS{IIP}, k = dimension(ds)) where {IIP}
    if IIP
        J = deepcopy(ds.J)
        tangentf = (du, u, p, t) -> begin
            uv = @view u[:, 1]
            ds.prob.f(view(du, :, 1), uv, p, t)
            ds.jacobian(J, uv, p, t)
            A_mul_B!((@view du[:, 2:end]), J, (@view u[:, 2:end]))
            nothing
        end
    else
        ws_index = SVector{k, Int}(2:(k+1)...)
        tangentf = (u, p, t) -> begin
            du = ds.prob.f(u[:, 1], p, t)
            J = ds.jacobian(u[:, 1], p, t)

            dW = J*u[:, ws_index]
            return hcat(du, dW)
        end
    end
    return tangentf
end

# for the case of autodiffed systems, a specialized version is created
# so that f! is not called twice in ForwardDiff
function create_tangent(ds::DS{true, true})
    J = deepcopy(ds.J)
    cfg = ForwardDiff.JacobianConfig(
        (du, u) -> (du, u, ds.prob.p, inittime(ds)),
        deepcopy(state(ds)), state(ds)
    )
    tangentf = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        ForwardDiff.jacobian!(
            J, (du, u) -> (du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
        )
        A_mul_B!((@view du[:, 2:end]), J, (@view u[:, 2:end]))
        nothing
    end
    return tangentf
end

# Create equations of motion of evolving states in parallel
function create_parallel(ds::DS{true}, states)
    st = [Vector(s) for s in states]
    L = length(st)
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            @inbounds ds.prob.f(du[i], u[i], p, t)
        end
    end
    return paralleleom, st
end

function create_parallel(ds::DS{false}, states)
    D = length(states[1])
    st = [SVector{D}(s) for s in states]
    L = length(st)
    # The following may be inneficient
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            @inbounds du[i] = ds.prob.f(u[i], p, t)
        end
    end
    return paralleleom, st
end

#######################################################################################
#                                     Docstrings                                      #
#######################################################################################
"""
    integrator(ds::DynamicalSystem [, u0]; diff_eq_kwargs) -> integ
Return a `DEIntegrator` object that can be used to evolve a system interactively
using `step!(integ [, Δt])`. Optionally specify an initial state `u0`.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function integrator end

"""
    tangent_integrator(ds::DynamicalSystem, Q0 | k::Int; u0, diff_eq_kwargs)
Return a `DEIntegrator` object that evolves in parallel both the system as well
as deviation vectors living on the tangent space.

`Q0` is a *matrix* whose columns are initial values for deviation vectors. If
instead of a matrix `Q0` an integer `k` is given, then `k` random orthonormal
vectors are choosen as initial conditions. You can also give as a keyword argument
a different initial state for the system `u0`.

See [`trajectory`](@ref) for `diff_eq_kwargs`.

## Description

If ``J`` is the jacobian of the system then the *tangent dynamics* are
the equations that evolve in parallel the system as well as
a deviation vector (or matrix) ``w``:
```math
\\begin{aligned}
\\dot{u} &= f(u, p, t) \\\\
\\dot{w} &= J(u, p, t) \\times w
\\end{aligned}
```
with ``f`` being the equations of motion and ``u`` the system state.
Similar equations hold for the discrete case.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function tangent_integrator end

"""
    parallel_integrator(ds::DynamicalSystem, states; diff_eq_kwargs) -> integ
Return a `DEIntegrator` object that can be used to evolve many `states` of
a system in parallel interactively using `step!(integ [, Δt])`.

*Warning* : Callbacks do not propagate into `parallel_integrator`.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function parallel_integrator end

"""
    trajectory(ds::DynamicalSystem, T [, u]; kwargs...) -> dataset

Return a dataset what will contain the trajectory of the sytem,
after evolving it for total time `T`, optionally starting from state `u`.
See [`Dataset`](@ref) for info on how to
manipulate this object.

For the discrete case, `T` is an integer and a `T×D` dataset is returned
(`D` is the system dimensionality). For the
continuous case, a `W×D` dataset is returned, with `W = length(t0:dt:T)` with
`t0:dt:T` representing the time vector (*not* returned).

## Keyword Arguments
* `dt` :  Time step of value output during the solving
  of the continuous system. For discrete systems it must be an integer. Defaults
  to `0.01` for continuous and `1` for discrete.
* `diff_eq_kwargs` : (only for continuous) A dictionary `Dict{Symbol, ANY}`
  of keyword arguments
  passed into the solvers of the [DifferentialEquations.jl](http://docs.juliadiffeq.org/latest/basics/common_solver_opts.html)
  package, for example `Dict(:abstol => 1e-9)`. If you want to specify a solver,
  do so by using the symbol `:solver`, e.g.:
  `Dict(:solver => DP5(), :maxiters => 1e9)`. This requires you to have been first
  `using OrdinaryDiffEq` to access the solvers. Defaults to
  `Dict(:solver => Vern9(), :abstol => 1e-9, :reltol => 1e-9)`.
"""
function trajectory end

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

"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(prob::DEProblem) = length(prob.u0)
state(prob::DEProblem) = prob.u0
state(integ::DEIntegrator) = integ.u
hascallback(prob::DEProblem) = :callback ∈ fieldnames(prob) && prob.callback != nothing
statetype(prob::DEProblem) = eltype(prob.u0)
systemtype(::ODEProblem) = "continuous"
systemtype(::DiscreteProblem) = "discrete"
inittime(prob::DEProblem) = prob.tspan[1]

safe_state_type(ds::DS{true}, u0) = Vector(u0)
safe_state_type(ds::DS{false}, u0) = SVector{length(u0)}(u0...)
safe_state_type(ds::DS{false}, u0::Number) = u0

safe_matrix_type(ds::DS{true}, Q::AbstractMatrix) = Matrix(Q)
function safe_matrix_type(ds::DS{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
safe_matrix_type(IIP::Bool, Q::AbstractMatrix) =
IIP ? Matrix(Q) : SMatrix{size(Q)...}(Q)
