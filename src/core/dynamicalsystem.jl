using LinearAlgebra, DiffEqBase, ForwardDiff, StaticArrays
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
can use known dynamic rule `f` (equations of motion) expect an instance of this type.

## Constructing a `DynamicalSystem`
```julia
DiscreteDynamicalSystem(f, state, p [, jacobian [, J0]]; t0::Int = 0)
ContinuousDynamicalSystem(f, state, p [, jacobian [, J0]]; t0 = 0.0)
```
with `f` a Julia function (see below).
`p` is a parameter container, which we highly suggest to be a mutable, concretely typed
container. Pass `nothing` as `p` if your system does not have parameters.

`t0`, `J0` allow you to choose the initial time and provide
an initialized Jacobian matrix. See `CDS_KWARGS` for the
default options used to evolve continuous systems (through `OrdinaryDiffEq`).

## Dynamic rule `f`
The are two "versions" for `DynamicalSystem`, depending on whether `f` is
in-place (iip) or out-of-place (oop).
Here is how to define them (1D systems are treated differently, see below):

* **oop** : `f` **must** be in the form `f(x, p, t) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an [`SVector`](http://juliaarrays.github.io/StaticArrays.jl/stable/pages/api.html#SVector-1)
  (from the [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) module)
  containing the next state/rate-of-change.
* **iip** : `f` **must** be in the form `f!(xnew, x, p, t)`
  which means that given a state `x::Vector` and some parameter container `p`,
  it writes in-place the new state/rate-of-change in `xnew`.

`t` stands for time (integer for discrete systems).
iip is suggested for big systems, whereas oop is suggested for small systems.
The break-even point at around 10 dimensions.

The constructor deduces automatically whether `f` is iip or oop. It is not
possible however to deduce whether the system is continuous or discrete just from `f`,
hence the 2 constructors.

### Jacobian
The optional argument `jacobian` for the constructors
is a *function* and (if given) must also be of the same form as the `eom`,
`jacobian(x, p, n) -> SMatrix`
for the out-of-place version and `jacobian!(Jnew, x, p, n)` for the in-place version.

The constructors also allow you to pass an initialized Jacobian matrix `J0`.
This is useful for large oop systems where only a few components of the Jacobian change
during the time evolution.

If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).
Even though `ForwardDiff` is very fast, depending on your exact system you might
gain significant speed-up by providing a hand-coded Jacobian and so we recommend it.

### Comment on 1-D
One dimensional discrete systems expect the state always as a pure number, `0.8` instead
of `SVector(0.8)`. For continuous systems, the state can be in-place/out-of-place as
in higher dimensions, however the derivative function must be always explicitly given.

## Interface to DifferentialEquations.jl
Continuous systems are solved using
[**DifferentialEquations.jl**](http://docs.juliadiffeq.org/latest/), by default
using the keyword arguments contained in the constant `CDS_KWARGS.`

The following two interfaces are provided:
```
ContinuousDynamicalSystem(prob::ODEProblem [, jacobian [, J0]])
ODEProblem(continuous_dynamical_system, tspan, args...)
```
where in the second case `args` stands for the
standard extra arguments of `ODEProblem`: `callback, mass_matrix`.

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
    ContinuousDynamicalSystem(prob::ODEProblem [, jacobian [, J0]])
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


DiffEqBase.isinplace(::DS{IIP}) where {IIP} = IIP
statetype(::DS{IIP, S}) where {IIP, S} = S
stateeltype(::DS{IIP, S}) where {IIP, S} = eltype(S)
isautodiff(::DS{IIP, S, D, F, P, JAC, JM, IAD}) where
{IIP, S, D, F, P, JAC, JM, IAD} = IAD

get_state(ds::DS) = ds.u0
DelayEmbeddings.dimension(ds::DS{IIP, S, D}) where {IIP, S, D} = D

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
    if ds == :ContinuousDynamicalSystem
        oneder = :(error("For 1D continuous systems, you need to explicitly give a derivative function."))
    else
        oneder = :(nothing)
    end

    @eval function $(ds)(eom, u0, p; t0 = 0.0)

        if !(typeof(u0) <: Union{AbstractVector, Number})
            throw(ArgumentError(
            "The state of a dynamical system *must* be <: AbstractVector/Number!"))
        end

        IIP = isinplace(eom, 4)
        s = safe_state_type(Val{IIP}(), u0)
        D = length(s)
        D == 1 && $(oneder)

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
    @nospecialize(f::F), ::Val{IIP}, s::S, p::P, t::T, ::Val{D}) where {F, IIP, S, P, T, D}
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
