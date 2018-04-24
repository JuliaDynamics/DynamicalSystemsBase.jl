using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, step!

export dimension, get_state, DynamicalSystem, jacobian
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export set_parameter!, step!, inittime
export trajectory
export integrator, tangent_integrator, parallel_integrator
export set_state!, get_state, get_deviations, set_deviations!

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################
"""
    DynamicalSystem

The central structure of **DynamicalSystems.jl**. All functions of the suite that
handle systems "analytically" (in the sense that they can use known equations of
motion) expect an instance of this type.

Contains a problem defining the system (field `prob`), the jacobian function
(field `jacobian`) and the initialized Jacobian matrix (field `J`).

## Constructing a `DynamicalSystem`
```julia
DiscreteDynamicalSystem(eom, state, p [, jacobian [, J0]]; t0::Int = 0)
ContinuousDynamicalSystem(eom, state, p [, jacobian [, J0]]; t0 = 0.0)
```
with `eom` the equations of motion function.
`p` is a parameter container, which we highly suggest to use a mutable object like
`Array`, [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl) or
a dictionary. Pass `nothing` in the place of `p` if your system does not have
parameters. With these constructors you also do not need to
provide some final time, since it is not used by **DynamicalSystems.jl** in any manner.

`t0`, `J0` allow you to choose the initial time and provide
an initialized Jacobian matrix.

Continuous system solvers use [**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/)
and by default are integrated with a 9th order Verner solver `Vern9()` with tolerances
`abstol = reltol = 1e-9`.

### Equations of motion
The are two "versions" for `DynamicalSystem`, depending on whether the
equations of motion (`eom`) are in-place (iip) or out-of-place (oop).
Here is how to define them:

* **oop** : The `eom` **must** be in the form `eom(x, p, t) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an [`SVector`](http://juliaarrays.github.io/StaticArrays.jl/stable/pages/api.html#SVector-1)
  (from the [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) module)
  containing the next state.
* **iip** : The `eom` **must** be in the form `eom!(xnew, x, p, t)`
  which means that given a state `x::Vector` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

`t` stands for time (integer for discrete systems).
iip is suggested for big systems, whereas oop is suggested for small systems.
The break-even point at around 100 dimensions, and for using functions that use the
tangent space (like e.g. `lyapunovs` or `gali`), the break-even
point is at around 10 dimensions.

The constructor deduces automatically whether `eom` is iip or oop. It is not
possible however to deduce whether the system is continuous or discrete just from the
equations of motion, hence the 2 constructors.

### Jacobian
The optional argument `jacobian` for the constructors
is a *function* and (if given) must also be of the same form as the `eom`,
`jacobian(x, p, n) -> SMatrix`
for the out-of-place version and `jacobian!(xnew, x, p, n)` for the in-place version.

If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).

### Getting a `Solution` struct
The continuous constructor creates a standard
[ODEProblem](http://docs.juliadiffeq.org/latest/types/ode_types.html) from
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/).

You can *always* take advantage of the full capabilities of
the `Solution` struct. Simply define
```julia
using DifferentialEquations
prob = continuousds.prob
prob2 = remake(prob1; tspan=(0.0,2.0))
sol = solve(prob2, Tsit5())
# do stuff with sol...
```
The line `remake...` is necessary because by default the `tspan` of all problems
ends at infinity. See the
[remake documentation](http://docs.juliadiffeq.org/latest/basics/problem.html)
for more info.

## Relevant Functions
[`trajectory`](@ref), [`jacobian`](@ref), [`dimension`](@ref),
[`set_parameter!`](@ref).
"""
abstract type DynamicalSystem{
        IIP,     # is in place , for dispatch purposes and clarity
        S,       # state type
        D,       # dimension
        F,       # equations of motion
        P,       # parameters
        JAC,     # jacobian
        JM,      # jacobian matrix
        IAD}     # is auto-differentiated
    # one-liner: {IIP, S, D, F, P, JAC, JM, IAD}
    # Subtypes of DynamicalSystem have fields:
    # 1. prob
    # 2. jacobian (function)
    # 3. J (matrix)  <- will allow Sparse implementation in the future
end

const DS = DynamicalSystem

# Type stability methods:
isinplace(ds::DS{IIP}) where {IIP} = IIP
statetype(ds::DS{IIP, S}) where {IIP, S} = S
stateeltype(ds::DS{IIP, S}) where {IIP, S} = eltype(S)

isautodiff(ds::DS{IIP, S, D, F, P, JAC, JM, IAD}) where
{IIP, S, D, F, P, JAC, JM, IAD} = IAD


"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(ds::DS{IIP, S, D}) where {IIP, S, D} = D
get_state(ds::DS) = ds.prob.u0
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

dimension(prob::DEProblem) = length(prob.u0)
hascallback(prob::ODEProblem) = prob.callback != nothing
inittime(prob::DEProblem) = prob.tspan[1]
inittime(ds::DS) = inittime(ds.prob)

safe_state_type(::Val{true}, u0) = u0
safe_state_type(::Val{false}, u0) = SVector{length(u0)}(u0...)
safe_state_type(::Val{false}, u0::SVector) = u0
safe_state_type(::Val{false}, u0::Number) = u0

safe_matrix_type(::Val{true}, Q::Matrix) = Q
safe_matrix_type(::Val{true}, Q::AbstractMatrix) = Matrix(Q)
function safe_matrix_type(::Val{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
save_matrix_type(::Val{false}, Q::SMatrix) = Q
safe_matrix_type(_, a::Number) = a

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
systemtype(::ODEProblem) = "continuous"
systemtype(something) = "discrete"
Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds.prob)*" dynamical system"

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(ds.jacobian)"

function Base.show(io::IO, ds::DS)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(get_state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    rpad(" jacobian: ", ps)*"$(jacobianstring(ds))\n"
    )
end

#######################################################################################
#                                    Jacobians                                        #
#######################################################################################
function create_jacobian(
    f::F, ::Val{IIP}, s::S, p::P, t::T, x::Val{D}) where {F, IIP, S, P, T, D}
    if IIP
        dum = deepcopy(s)
        cfg = ForwardDiff.JacobianConfig(
        (y, x) -> f(y, x, p, t), dum, s)
        jac = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> f(y, x, p, t),
        dum, u, cfg, Val{false}())
        return jac
    else
        if x == Val{1}()
            return jac = (u, p, t) -> ForwardDiff.derivative((x) -> f(x, p, t), u)
        else
            # SVector methods do *not* use the config
            # cfg = ForwardDiff.JacobianConfig(
            #     (x) -> prob.f(x, prob.p, prob.tspan[1]), prob.u0)
            return jac = (u, p, t) ->
            ForwardDiff.jacobian((x) -> f(x, p, t), u #=, cfg=#)
        end
    end
end

# get_J function is defined at individual files

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

function get_J(prob, jacob::JAC) where {JAC}
    D = length(prob.u0)
    if isinplace(prob)
        J = similar(prob.u0, (D,D))
        jacob(J, prob.u0, prob.p, inittime(prob))
    else
        J = jacob(prob.u0, prob.p, inittime(prob))
    end
    return J
end
function get_J(jacob::JAC, u, p, t) where {JAC}
    D = length(u)
    if isinplace(jacob, 4)
        J = similar(u, (D,D))
        jacob(J, u, p, t)
    else
        J = jacob(u, p, t)
    end
    return J
end

#######################################################################################
#                                 Tanget Dynamics                                     #
#######################################################################################
# IIP Tangent Space dynamics
function create_tangent(f::F, jacobian::JAC, J::JM,
    ::Val{true}, ::Val{k}) where {F, JAC, JM, k}
    J = deepcopy(J)
    tangentf = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        f(view(du, :, 1), uv, p, t)
        jacobian(J, uv, p, t)
        A_mul_B!((@view du[:, 2:(k+1)]), J, (@view u[:, 2:(k+1)]))
        nothing
    end
    return tangentf
end
# OOP Tangent Space dynamics
function create_tangent(f::F, jacobian::JAC, J::JM,
    ::Val{false}, ::Val{k}) where {F, JAC, JM, k}

    ws_index = SVector{k, Int}(2:(k+1)...)
    tangentf = TangentOOP(f, jacobian, ws_index)
    return tangentf
end
struct TangentOOP{F, JAC, k}
    f::F
    jacobian::JAC
    ws::SVector{k, Int}
end
@inline function (tan::TangentOOP)(u, p, t)
    du = tan.f(u[:, 1], p, t)
    J = tan.jacobian(u[:, 1], p, t)
    dW = J*u[:, tan.ws]
    return hcat(du, dW)
end

# for the case of autodiffed systems, a specialized version is created
# so that f! is not called twice in ForwardDiff
function create_tangent_iad(f::F, J::JM, u, p, t, ::Val{k}) where {F, JM, k}
    let
        J = deepcopy(J)
        cfg = ForwardDiff.JacobianConfig(
            (du, u) -> f(du, u, p, p), deepcopy(u), deepcopy(u)
        )
        tangentf = (du, u, p, t) -> begin
            uv = @view u[:, 1]
            ForwardDiff.jacobian!(
                J, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
            )
            A_mul_B!((@view du[:, 2:k+1]), J, (@view u[:, 2:k+1]))
            nothing
        end
        return tangentf
    end
end

#######################################################################################
#                                Parallel Dynamics                                    #
#######################################################################################
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
    D = dimension(ds)
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
Return an integrator object that can be used to evolve a system interactively
using `step!(integ [, Δt])`. Optionally specify an initial state `u0`.

The state of this integrator is a vector.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function integrator end

"""
    get_state(ds::DynamicalSystem)
Return the state of `ds`.

    get_state(integ [, i::Int = 1])
Return the state of the integrator, in the sense of the state of the dynamical system.

If the integrator is a [`parallel_integrator`](@ref), passing `i` will return
the `i`-th state. The function also correctly returns the true state of the system
for tangent integrators.
"""
get_state(integ) = integ.u

"""
    set_state!(integ, u [, i::Int = 1])
Set the state of the integrator to `u`, in the sense of the state of the
dynamical system. Works for any integrator (normal, tangent, parallel).

For parallel integrator, you can choose which state to set (using `i`).
"""
set_state!(integ, u) = (integ.u = u)

"""
    tangent_integrator(ds::DynamicalSystem, Q0 | k::Int; u0, t0, diff_eq_kwargs)
Return an integrator object that evolves in parallel both the system as well
as deviation vectors living on the tangent space.

`Q0` is a *matrix* whose columns are initial values for deviation vectors. If
instead of a matrix `Q0` an integer `k` is given, then `k` random orthonormal
vectors are choosen as initial conditions.
You can also give as a keyword argument
a different initial state or time `u0, t0`.

The state of this integrator is a matrix with the first column the system state
and all other columns being deviation vectors.

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
"""
function tangent_integrator end

"""
    get_deviations(tang_integ)
Return the deviation vectors of the [`tangent_integrator`](@ref) in a form
of a matrix with columns the vectors.
"""
function get_deviations end

"""
    set_deviations!(tang_integ, Q)
Set the deviation vectors of the [`tangent_integrator`](@ref) to `Q`, which must
be a matrix with each column being a deviation vector.
"""
function set_deviations end

"""
    parallel_integrator(ds::DynamicalSystem, states; diff_eq_kwargs)
Return an integrator object that can be used to evolve many `states` of
a system in parallel at the *exact same times*, using `step!(integ [, Δt])`.

The states of this integrator are a vector of vectors, each one being an actual
state of the dynamical system.
Only for the case of in-place continuous systems, the integrator propagates a matrix
with each column being a state, because at the moment DifferentialEquations.jl does
not support `Vector[Vector]` as state.

See [`trajectory`](@ref) for `diff_eq_kwargs`.
"""
function parallel_integrator end

"""
    trajectory(ds::DynamicalSystem, T [, u]; kwargs...) -> dataset

Return a dataset that will contain the trajectory of the system,
after evolving it for total time `T`, optionally starting from state `u`.
See [`Dataset`](@ref) for info on how to use this object.

A `W×D` dataset is returned, with `W = length(t0:dt:T)` with
`t0:dt:T` representing the time vector (*not* returned) and `D` the system dimension.
For discrete systems both `T` and `dt` must be integers.

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
  `Dict(:solver => Vern9(), :abstol => 1e-9, :reltol => 1e-9)`, i.e. a 9th order
  Verner algorithm.
"""
function trajectory end
