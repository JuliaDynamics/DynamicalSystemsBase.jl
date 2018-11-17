using LinearAlgebra, DiffEqBase, ForwardDiff, StaticArrays
import DiffEqBase: isinplace, reinit!
using SparseArrays

export dimension, get_state, DynamicalSystem
export ContinuousDynamicalSystem, DiscreteDynamicalSystem
export set_parameter!, step!
export trajectory, jacobian
export integrator, tangent_integrator, parallel_integrator
export set_state!, get_state, get_deviations, set_deviations!

#####################################################################################
#                                  DynamicalSystem                                  #
#####################################################################################
"""
    DynamicalSystem

The central structure of **DynamicalSystems.jl**. All functions of the suite that
can use known equations of motion expect an instance of this type.

## Constructing a `DynamicalSystem`
```julia
DiscreteDynamicalSystem(eom, state, p [, jacobian [, J0]]; t0::Int = 0)
ContinuousDynamicalSystem(eom, state, p [, jacobian [, J0]]; t0 = 0.0)
```
with `eom` the equations of motion function (see below).
`p` is a parameter container, which we highly suggest to use a mutable object like
`Array`, [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl) or
a dictionary. Pass `nothing` in the place of `p` if your system does not have
parameters.

`t0`, `J0` allow you to choose the initial time and provide
an initialized Jacobian matrix. See `CDS_KWARGS` for the
default options used to evolve continuous systems (through `OrdinaryDiffEq`).

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
It is **heavily** advised to provide a Jacobian function, as it gives *multiple*
orders of magnitude speedup.

### Interface to DifferentialEquations.jl
Continuous systems are solved using
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/).
The following two interfaces are provided:
```
ContinuousDynamicalSystem(prob::ODEProblem [, jacobian [, J0]])
ODEProblem(continuous_dynamical_system, tspan, args...)
```
where in the second case `args` stands for the
[standard extra arguments](http://docs.juliadiffeq.org/latest/types/ode_types.html#Constructors-1)
of `ODEProblem`: `callback, mass_matrix`.

If you want to use callbacks with [`tangent_integrator`](@ref) or
[`parallel_integrator`](@ref), then invoke them with extra arguments
as shown in the [Advanced Documentation](https://juliadynamics.github.io/DynamicalSystems.jl/latest/advanced/).

## Relevant Functions
[`trajectory`](@ref), [`set_parameter!`](@ref).
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
    # 1. f
    # 2. u0
    # 3. p
    # 4. t0
    # 5. jacobian (function)
    # 6. J (matrix)
end
const DS = DynamicalSystem

"""
    ContinuousDynamicalSystem(eom, state, p [, jacobian [, J]]; t0 = 0.0)
A `DynamicalSystem` restricted to continuous-time systems (also called *ODEs*).
"""
struct ContinuousDynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD} <:
                 DynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD}
    f::F
    u0::S
    p::P
    t0::eltype(S)
    jacobian::JAC
    J::JM
end
const CDS = ContinuousDynamicalSystem
systemtype(::CDS) = "continuous"

"""
    DiscreteDynamicalSystem(eom, state, p [, jacobian [, J]]; t0::Int = 0)
A `DynamicalSystem` restricted to discrete-time systems (also called *maps*).
"""
struct DiscreteDynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD} <:
               DynamicalSystem{IIP, S, D, F, P, JAC, JM, IAD}
    f::F
    u0::S
    p::P
    t0::Int
    jacobian::JAC
    J::JM
end
const DDS = DiscreteDynamicalSystem
systemtype(::DDS) = "discrete"


isinplace(::DS{IIP}) where {IIP} = IIP
statetype(::DS{IIP, S}) where {IIP, S} = S
stateeltype(::DS{IIP, S}) where {IIP, S} = eltype(S)
isautodiff(::DS{IIP, S, D, F, P, JAC, JM, IAD}) where
{IIP, S, D, F, P, JAC, JM, IAD} = IAD

get_state(ds::DS) = ds.u0



"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(ds::DS{IIP, S, D}) where {IIP, S, D} = D

"""
    set_parameter!(ds::DynamicalSystem, index, value)
    set_parameter!(ds::DynamicalSystem, values)
Change one or many parameters of the system
by setting `p[index] = value` in the first case
and `p .= values` in the second.

The same function also works for any integrator.
"""
set_parameter!(ds, index, value) = (ds.p[index] = value)
set_parameter!(ds, values) = (ds.p .= values)


#####################################################################################
#                           State types enforcing                                   #
#####################################################################################
safe_state_type(::Val{true}, u0) = Vector(u0)
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
#                                    Constructors                                   #
#####################################################################################
for ds in (:ContinuousDynamicalSystem, :DiscreteDynamicalSystem)

    if ds == :ContinuousDynamicalSystem
        @eval timetype(::Type{$(ds)}, s) = eltype(s)
    else
        @eval timetype(::Type{$(ds)}, s) = Int
    end

    # Main Constructor (with Jacobian and j0)
    @eval function $(ds)(
        eom::F, u0, p::P, j::JAC, j0 = nothing;
        t0 = 0.0, IAD = false, IIP = isinplace(eom, 4)) where {F, P, JAC}

        if !(typeof(u0) <: Union{AbstractVector, Number})
            throw(ArgumentError(
            "The state of a dynamical system *must* be <: AbstractVector/Number!"))
        end

        s = safe_state_type(Val{IIP}(), u0)
        S = typeof(s)
        D = length(s)

        IIP || typeof(eom(s, p, t0)) <: Union{SVector, Number} ||
        throw(ArgumentError("Equations of motion must return an `SVector` "*
        "or number for out-of-place form!"))

        J = j0 != nothing ? j0 : get_J(j, s, p, timetype($(ds), s)(t0), IIP)
        JM = typeof(J)

        return $(ds){
            IIP, S, D, F, P, JAC, JM, IAD}(eom, u0, p, t0, j, J)
    end

    # Without Jacobian
    @eval function $(ds)(eom, u0, p; t0 = 0.0)

        if !(typeof(u0) <: Union{AbstractVector, Number})
            throw(ArgumentError(
            "The state of a dynamical system *must* be <: AbstractVector/Number!"))
        end

        IIP = isinplace(eom, 4)
        s = safe_state_type(Val{IIP}(), u0)
        D = length(s)

        t = timetype($(ds), s)(t0)
        j = create_jacobian(eom, Val{IIP}(), s, p, t, Val{D}())
        J = get_J(j, s, p, t, IIP)

        return $(ds)(eom, s, p, j, J; t0 = t0, IAD = true, IIP = IIP)
    end
end

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds)*" dynamical system"

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(eomstring(ds.jacobian))"
eomstring(f::Function) = nameof(f)
eomstring(f) = nameof(typeof(f))

paramname(p::AbstractArray) = string(p)
paramname(p::Nothing) = repr(p)
paramname(p) = nameof(typeof(p))

# Credit to Sebastian Pfitzner
function printlimited(io, x; Δx = 0, Δy = 0)
    sz = displaysize(io)
    io2 = IOBuffer(); ctx = IOContext(io2, :limit => true, :compact => true,
    :displaysize => (sz[1]-Δy, sz[2]-Δx))
    Base.print_array(ctx, x)
    s = String(take!(io2))
    s = replace(s[2:end], "  " => ", ")
    Base.print(io, "["*s*"]")
end

printlimited(io, x::Number; Δx = 0, Δy = 0) = print(io, x)

function Base.show(io::IO, ds::DS)
    ps = 14
    text = summary(ds)
    u0 = get_state(ds)'

    ctx = IOContext(io, :limit => true, :compact => true, :displaysize => (10,50))

    println(io, text)
    prefix = rpad(" state: ", ps)
    print(io, prefix); printlimited(io, u0, Δx = length(prefix)); print(io, "\n")
    println(io,  rpad(" e.o.m.: ", ps),     eomstring(ds.f))
    println(io,  rpad(" in-place? ", ps),   isinplace(ds))
    println(io,  rpad(" jacobian: ", ps),   jacobianstring(ds)),
    print(io,    rpad(" parameters: ", ps), paramname(ds.p))
end

#######################################################################################
#                                    Jacobians                                        #
#######################################################################################
function create_jacobian(
    f::F, ::Val{IIP}, s::S, p::P, t::T, ::Val{D}) where {F, IIP, S, P, T, D}
    if IIP
        dum = deepcopy(s)
        cfg = ForwardDiff.JacobianConfig(
        (y, x) -> f(y, x, p, t), dum, s)
        jac = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> f(y, x, p, t),
        dum, u, cfg, Val{false}())
        return jac
    else
        if D == 1
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

function get_J(jacob, u0, p, t0, iip) where {JAC}
    D = length(u0)
    if iip
        J = similar(u0, (D,D))
        jacob(J, u0, p, t0)
    else
        J = jacob(u0, p, t0)
    end
    return J
end


"""
    jacobian(ds::DynamicalSystem, u = ds.u0, t = ds.t0)
Return the jacobian of the system at `u`, at `t`.
"""
function jacobian(ds::DS{true}, u = ds.u0, t = ds.t0)
    J = similar(ds.J)
    ds.jacobian(J, u, ds.p, t)
    return J
end
jacobian(ds::DS{false}, u = ds.u0, t = ds.t0) =
ds.jacobian(u, ds.p, t)


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
        mul!((@view du[:, 2:(k+1)]), J, (@view u[:, 2:(k+1)]))
        nothing
    end
    return tangentf
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
            mul!((@view du[:, 2:k+1]), J, (@view u[:, 2:k+1]))
            nothing
        end
        return tangentf
    end
end



# OOP Tangent Space dynamics
function create_tangent(f::F, jacobian::JAC, J::JM,
    ::Val{false}, ::Val{k}) where {F, JAC, JM, k}

    ws_index = SVector{k, Int}(2:(k+1)...)
    tangentf = TangentOOP{F, JAC, k}(f, jacobian, ws_index)
    return tangentf
end
struct TangentOOP{F, JAC, k} <: Function
    f::F
    jacobian::JAC
    ws::SVector{k, Int}
end
function (tan::TangentOOP)(u, p, t)
    @inbounds s = u[:, 1]
    du = tan.f(s, p, t)
    J = tan.jacobian(s, p, t)
    @inbounds dW = J*u[:, tan.ws]
    return hcat(du, dW)
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
            @inbounds ds.f(du[i], u[i], p, t)
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
            @inbounds du[i] = ds.f(u[i], p, t)
        end
    end
    return paralleleom, st
end

#######################################################################################
#                                     Docstrings                                      #
#######################################################################################
"""
    integrator(ds::DynamicalSystem [, u0]; diffeq...) -> integ
Return an integrator object that can be used to evolve a system interactively
using `step!(integ [, Δt])`. Optionally specify an initial state `u0`.

The state of this integrator is a vector.

* `diffeq...` are keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
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

Automatically does `u_modified!(integ, true)`.
"""
set_state!(integ, u) = (integ.u = u; u_modified!(integ, true))

"""
    tangent_integrator(ds::DynamicalSystem, Q0 | k::Int; kwargs...)
Return an integrator object that evolves in parallel both the system as well
as deviation vectors living on the tangent space.

`Q0` is a *matrix* whose columns are initial values for deviation vectors. If
instead of a matrix `Q0` an integer `k` is given, then `k` random orthonormal
vectors are choosen as initial conditions.

## Keyword Arguments
* `u0` : Optional different initial state.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

It is *heavily* advised to use the functions [`get_state`](@ref), [`get_deviations`](@ref),
[`set_state!`](@ref), [`set_deviations!`](@ref) to manipulate the integrator.

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

Automatically does `u_modified!(tang_integ, true)`.
"""
function set_deviations! end

"""
    parallel_integrator(ds::DynamicalSystem, states; kwargs...)
Return an integrator object that can be used to evolve many `states` of
a system in parallel at the *exact same times*, using `step!(integ [, Δt])`.

`states` are expected as vectors of vectors.

## Keyword Arguments
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  See [`trajectory`](@ref) for examples. Only valid for continuous systems.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

It is *heavily* advised to use the functions [`get_state`](@ref) and
[`set_state!`](@ref) to manipulate the integrator. Provide `i` as a second
argument to change the `i`-th state.
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
* `Ttr` : Transient time to evolve the initial state before starting saving states.
* `diffeq...` : Keyword arguments propagated into `init` of DifferentialEquations.jl.
  For example `abstol = 1e-9`.  Only valid for continuous systems.
  If you want to specify a solver, do so by using the name `alg`, e.g.:
  `alg = Tsit5(), maxiters = 1000`. This requires you to have been first
  `using OrdinaryDiffEq` to access the solvers. See
  `DynamicalSystemsBase.CDS_KWARGS` for default values.
  These keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).
  Using a `SavingCallback` with `trajectory` will lead to unexpected behavior!

"""
function trajectory end
