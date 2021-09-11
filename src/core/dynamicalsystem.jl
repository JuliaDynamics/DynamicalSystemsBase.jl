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

A central structure of **DynamicalSystems.jl**. All functions of the suite that
can use known dynamic rule `f` (also called equations of motion or vector field)
expect an instance of this type.

## Constructing a `DynamicalSystem`
```julia
DiscreteDynamicalSystem(f, state, p [, jacobian [, J0]]; t0::Int = 0)
ContinuousDynamicalSystem(f, state, p [, jacobian [, J0]]; t0 = 0.0)
```
with `f` a Julia function (see below).
`p` is a parameter container, which we highly suggest to be a mutable, concretely typed
container. Pass `nothing` as `p` if your system does not have parameters.

`t0`, `J0` allow you to choose the initial time and provide
an initialized Jacobian matrix. Continuous systems are evolved via the solvers of 
DifferentialEquations.jl, see `CDS_KWARGS` for the default options and the discussion
in [`trajectory`](@ref).

## Dynamic rule `f`
The are two "versions" for `DynamicalSystem`, depending on whether `f` is
in-place (iip) or out-of-place (oop).
Here is how to define them (1D systems are treated differently, see below):

* **oop** : `f` **must** be in the form `f(x, p, t) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector`
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

!!! note "Autonomous vs non-autonomous systems"
    Whether the dynamical system is autonomous (`f` doesn't depend on time) or not, it is
    still necessary to include `t` as an argument of `f`.
    While for some methods of DynamicalSystems.jl time-dependence is okay, the theoretical 
    foundation of many functions of the library only makes sense with autonomous systems.
    If you use a non-autonomous system, it is your duty to know for which functions this is okay.

### Jacobian
The optional argument `jacobian` for the constructors
is a *function* and (if given) must also be of the same form as `f`,
`jacobian(x, p, n) -> SMatrix`
for the out-of-place version and `jacobian!(J, x, p, n)` for the in-place version.

The constructors also allows you to pass an initialized Jacobian matrix `J0`.
This is useful for large iip systems where only a few components of the Jacobian change
during the time evolution. `J0` can have sparse structure for iip.

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
    ContinuousDynamicalSystem(f, state, p [, jacobian [, J]]; t0 = 0.0)
    ContinuousDynamicalSystem(prob::ODEProblem [, jacobian [, J0]])
A `DynamicalSystem` restricted to continuous-time systems (also called *ODEs*).
See the documentation of [`DynamicalSystem`](@ref) for more details.
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
    DiscreteDynamicalSystem(f, state, p [, jacobian [, J]]; t0::Int = 0)
A `DynamicalSystem` restricted to discrete-time systems (also called *maps*).
See the documentation of [`DynamicalSystem`](@ref) for more details.
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
isautodiff(::DS{IIP, S, D, F, P, JAC, JM, IAD}) where {IIP, S, D, F, P, JAC, JM, IAD} = IAD

get_state(ds::DS) = ds.u0
DelayEmbeddings.dimension(ds::DS{IIP, S, D}) where {IIP, S, D} = D

"""
    set_parameter!(ds::DynamicalSystem, index, value)
Change a parameter of the system given the index it has in the parameter container `p`
and the `value` to set it to. This function works for both array/dictionary containers
as well as composite types. In the latter case `index` needs to be a `Symbol`.


    set_parameter!(ds::DynamicalSystem, values)
In this case do `p .= values` (which only works for abstract array `p`).

The same function also works for any integrator.
"""
function set_parameter!(ds, index, value) 
    if ds.p isa Union{AbstractArray, AbstractDict}
        setindex!(ds.p, value, index)
    else
        setproperty!(ds.p, index, value)
    end
end

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
            f::F, u0, p::P, j::JAC, j0 = nothing;
            t0 = 0.0, IAD = false, IIP = isinplace(f, 4)
        ) where {F, P, JAC}

        if !(typeof(u0) <: Union{AbstractVector, Number})
            throw(ArgumentError(
                "The state of a dynamical system *must* be <: AbstractVector/Number!"
            ))
        end
        s = safe_state_type(Val{IIP}(), u0)
        S = typeof(s)
        D = length(s)

        if !(IIP || typeof(f(s, p, t0)) <: Union{SVector, Number})
            throw(ArgumentError(
                "Dynamic rule `f` must return an `SVector` "*
                "or pure number for out-of-place form!"
            ))
        end

        J = j0 !== nothing ? j0 : get_J(j, s, p, t0, IIP)
        JM = typeof(J)
        return $(ds){IIP, S, D, F, P, JAC, JM, IAD}(f, s, p, t0, j, J)
    end

    # Constructor without Jacobian (uses ForwardDiff then)
    if ds == :ContinuousDynamicalSystem
        oneder = :(error("For 1D continuous systems, you need to explicitly "*
            "give a derivative function."))
    else
        oneder = :(nothing)
    end

    @eval function $(ds)(f, u0, p; t0 = 0.0)
        if !(typeof(u0) <: Union{AbstractVector, Number})
            throw(ArgumentError(
            "The state of a dynamical system *must* be <: AbstractVector/Number!"))
        end
        IIP = isinplace(f, 4)
        s = safe_state_type(Val{IIP}(), u0)
        D = length(s)
        D == 1 && $(oneder)
        t = timetype($(ds), s)(t0)
        j = create_jacobian(f, Val{IIP}(), s, p, t, Val{D}())
        J = get_J(j, s, p, t, IIP)
        return $(ds)(f, s, p, j, J; t0 = t0, IAD = true, IIP = IIP)
    end
end

#######################################################################################
#                                    Jacobians                                        #
#######################################################################################
function create_jacobian(
        @nospecialize(f::F), ::Val{IIP}, s::S, p::P, t::T, ::Val{D}
    ) where {F, IIP, S, P, T, D}
    if IIP
        dum = deepcopy(s)
        inplace_f_2args = (y, x) -> f(y, x, p, t)
        cfg = ForwardDiff.JacobianConfig(inplace_f_2args, dum, s)
        jac! = (J, u, p, t) -> ForwardDiff.jacobian!(
            J, inplace_f_2args, dum, u, cfg, Val{false}()
        )
        return jac!
    else
        if D == 1
            return jac = (u, p, t) -> ForwardDiff.derivative((x) -> f(x, p, t), u)
        else
            # SVector methods do *not* use the config
            return jac = (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
        end
    end
end

function get_J(jacob, u0, p, t0, iip)
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
    jacobian(ds::DynamicalSystem, u = ds.u0, p = ds.p, t = ds.t0)
Return the jacobian of the system, always as a new matrix.
"""
function jacobian(ds::DS{true}, u = ds.u0, p = ds.p, t = ds.t0)
    J = similar(ds.J)
    ds.jacobian(J, u, p, t)
    return J
end
jacobian(ds::DS{false}, u = ds.u0, p = ds.p, t = ds.t0) = ds.jacobian(u, p, t)

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds)*" dynamical system"

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(eomstring(ds.jacobian))"
eomstring(f::Function) = nameof(f)
eomstring(f) = nameof(typeof(f))

# Credit to Sebastian Pfitzner
function printlimited(io, x; Δx = 0, Δy = 0)
    sz = displaysize(io)
    io2 = IOBuffer(); ctx = IOContext(io2, :limit => true, :compact => true,
    :displaysize => (sz[1]-Δy, sz[2]-Δx))
    if x isa AbstractArray
        Base.print_array(ctx, x)
        s = String(take!(io2))
        s = replace(s[2:end], "  " => ", ")
        Base.print(io, "["*s*"]")
    else
        Base.print(ctx, x)
        s = String(take!(io2))
        Base.print(io, s)
    end
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
    println(io,  rpad(" rule f: ", ps),     eomstring(ds.f))
    println(io,  rpad(" in-place? ", ps),   isinplace(ds))
    println(io,  rpad(" jacobian: ", ps),   jacobianstring(ds))
    print(io,    rpad(" parameters: ", ps))
    printlimited(io, printable(ds.p), Δx = length(prefix), Δy = 10)
end

printable(p::AbstractArray) = p'
printable(p::Nothing) = "nothing"
printable(p) = p

