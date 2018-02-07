using StaticArrays, ForwardDiff
import DiffEqBase: isinplace
import Base: eltype

export DiscreteDynamicalSystem, DDS, DiscreteProblem, DynamicalSystem
export state, jacobian, isinplace, dimension, statetype, state
export set_state!, set_parameter!, TangentEvolver, ParallelEvolver
export reform!
export evolve, evolve!
export trajectory
export DynamicalSystem

abstract type DynamicalSystem end

#####################################################################################
#                                 Discrete Problem                                  #
#####################################################################################
mutable struct DiscreteProblem{IIP, D, T, S, F, P}
    u0::S # initial state
    f::F  # eom, but same syntax as ODEProblem
    p::P  # parameter container
    dummy::S
end

"""
    DiscreteProblem(u0, eom, p = nothing) <: DynamicalSystem
Fundamental structure describing a discrete dynamical law.

## Fields
* `u0` : Initial state.
* `eom` : Function containing the equations of motion (EOM), ``u_{n+1} = f(u_n;p)``.
* `p` : parameter container. Don't pass anything if the dynamical system
  does not have parameters, otherwise we highly suggest to use
  a subtype of `Array` or
  [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl).

## Description
The are two "versions" for `DiscreteProblem`, depending on whether the
EOM are in-place (iip) or out-of-place (oop).
Here is how to define them:

* **iip** : The EOM **must** be in the form `eom(x, p) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector` containing the next state.
* **oop** : The EOM **must** be in the form `eom!(xnew, x, p)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

iip is suggested for big systems, whereas oop is suggested for small systems.

The constructor deduces automatically whether the EOM are iip or oop.
Use the function [`isinplace`](@ref) to test whether a given `DiscreteProblem`
operates in-place or out-of-place.

Pass a `DiscreteProblem` into the [`DiscreteDynamicalSystem`](@ref) constructor
to use the features of **DynamicalSystems.jl**.

## Related Functions
[`state`](@ref), [`set_state!`](@ref),
[`set_parameter!`](@ref),
[`ParallelEvolver`](@ref), [`TangentEvolver`](@ref).
"""
function DiscreteProblem(s, eom::F, p::P = nothing) where {F, P}
    D = length(s)
    T = eltype(s)
    iip = isinplace(eom, 3)
    iip || D != 1 && (@assert typeof(eom(s, p)) <: SVector)
    iip && (x = deepcopy(s); eom(x, s, p); @assert x!=s)
    u = iip ? Vector(s) : (D == 1 ? s : SVector{D}(s))
    S = typeof(u)
    DiscreteProblem{iip, D, T, S, F, P}(u, eom, p, deepcopy(u))
end

"""
    isinplace(ds) -> Bool
Return `true` if the system operates in-place.
"""
isinplace(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = IIP

"""
    dimension(ds::DynamicalSystem) -> D
Return the dimension of the system.
"""
dimension(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = D
eltype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = T
statetype(::DiscreteProblem{IIP, D, T, S, F, P}) where {IIP, D, T, S, F, P} = S

"""
    state(ds::DynamicalSystem) -> state
Return the current state of the system.
"""
state(prob::DiscreteProblem) = prob.u0

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
    set_state!(ds::DynamicalSystem, unew)
Set the state of the system to `unew`.
"""
set_state!(prob::DiscreteProblem{true}, xnew) = (prob.u0 .= xnew)
set_state!(prob::DiscreteProblem{false}, xnew) = (prob.u0 = xnew)


#####################################################################################
#                                 Time-Evolution                                    #
#####################################################################################
#= Methods necessary:
evolve(ds [, N] [, u])
evolve!([u], ds, N)
=#
"""
    evolve(ds::DynamicalSystem, T [, u0]; diff_eq_kwargs = Dict())
Evolve `u0` (or `state(ds)` if not given) for total time `T`
and return the `final_state`.

For discrete systems `T` corresponds to steps and
thus it must be integer. For continuous systems `T` can also be a tuple (for `tspan`).

`evolve` *does not store* any information about intermediate steps.
Use [`trajectory`](@ref) if you want to produce a trajectory of the system.
See also `[`trajectory`](@ref)` for using `diff_eq_kwargs`.
"""
evolve(prob::DiscreteProblem{true}, u = state(prob)) =
(prob.f(prob.dummy, u, prob.p); prob.dummy)
evolve(prob::DiscreteProblem{false}, u = state(prob)) = prob.f(u, prob.p)

function evolve(prob::DiscreteProblem{true}, N::Int, u = state(prob))
    D = dimension(prob)
    u0 = SVector{D}(u)
    prob.dummy .= u
    for i in 1:N
        prob.f(prob.u0, prob.dummy, prob.p)
        prob.dummy .= u
    end
    uret = SVector{D}(prob.u0)
    prob.u0 .= u0
    return Vector(uret)
end

function evolve(prob::DiscreteProblem{false}, N::Int, u0 = state(prob))
    for i in 1:N
        u0 = prob.f(u0, prob.p)
    end
    return u0
end

"""
    evolve!(ds::DynamicalSystem, T; diff_eq_kwargs = Dict())
Same as [`evolve`](@ref) but updates the system's state (in-place) with the
final state. See [`trajectory`](@ref) for `diff_eq_kwargs`.

*Note* - `evolve!` is non-allocating for discrete systems.

```julia
evolve!(pe::ParallelEvolver, N = 1)
```
Evolve `pe` for `N` steps in-place.
"""
evolve!(u, prob::DiscreteProblem{true}) =
(prob.dummy .= u; prob.f(u, prob.dummy, prob.p))
function evolve!(u, prob::DiscreteProblem{true}, N::Int)
    for i in 1:N
        prob.dummy .= u
        prob.f(u, prob.dummy, prob.p)
    end
    return
end
evolve!(prob::DiscreteProblem{true}) =
(prob.dummy .= prob.u0; prob.f(prob.u0, prob.dummy, prob.p))
evolve!(prob::DiscreteProblem{true}, N::Int) = evolve!(prob.u0, prob, N)

evolve!(u, prob::DiscreteProblem{false}) = (u .= prob.f(u, prob.p))
evolve!(u, prob::DiscreteProblem{false}, N::Int) = (u .= evolve(prob, N, u))
evolve!(prob::DiscreteProblem{false}, N::Int = 1) = (prob.u0 = evolve(prob, N))


"""
```julia
trajectory(ds::DynamicalSystem, T; kwargs...) -> dataset
```
Return a dataset what will contain the trajectory of the sytem,
after evolving it for time `T`. See [`Dataset`](@ref) for info on how to
manipulate this object (for 1D systems a `Vector` is returned).

For the discrete case, `T` is an integer and a `T×D` dataset is returned
(`D` is the system dimensionality). For the
continuous case, a `W×D` dataset is returned, with `W = length(0:dt:T)` with
`0:dt:T` representing the time vector (*not* returned).
## Keyword Arguments
* `dt = 0.05` : (only for continuous) Time step of value output during the solving
  of the continuous system.
* `diff_eq_kwargs = Dict()` : (only for continuous) A dictionary `Dict{Symbol, ANY}`
  of keyword arguments
  passed into the `solve` of the `DifferentialEquations.jl` package,
  for example `Dict(:abstol => 1e-9)`. If you want to specify a solver,
  do so by using the symbol `:solver`, e.g.:
  `Dict(:solver => DP5(), :maxiters => 1e9)`. This requires you to have been first
  `using OrdinaryDiffEq` to access the solvers.
"""
function trajectory(prob::DiscreteProblem{true}, N::Int, u = state(prob))
    SV = SVector{dimension(prob), eltype(u)}
    f! = prob.f
    ts = Vector{SV}(N)
    ts[1] = SV(u)
    for i in 2:N
        prob.dummy .= ts[i-1]
        f!(prob.u0, prob.dummy, prob.p)
        ts[i] = SV(prob.u0)
    end
    prob.u0 .= ts[1]
    return Dataset(ts)
end

function trajectory(prob::DiscreteProblem{false}, N::Int, st = state(prob))
    SV = SVector{dimension(prob), eltype(st)}
    ts = Vector{SV}(N)
    ts[1] = st
    f = prob.f
    for i in 2:N
        st = f(st, prob.p)
        ts[i] = st
    end
    return Dataset(ts)
end

function trajectory(prob::DiscreteProblem{false, 1}, N::Int, st = state(prob))
    T = eltype(prob)
    ts = Vector{T}(N)
    ts[1] = st
    f = prob.f
    for i in 2:N
        st = f(st, prob.p)
        ts[i] = st
    end
    return ts
end


#####################################################################################
#                               Parallel Evolvers                                   #
#####################################################################################
"""
    ParallelEvolver(prob::DiscreteProblem, states) -> pe
Return a structure `pe` that evolves in parallel many `states`
according to the same equations of motion.
The `states` must be a `Vector` of `Vector` or `SVector`.

Use [`evolve!`](@ref)`(pe, N)` to evolve all states for `N` steps.

Use [`set_state!`](@ref)`(pe, x, k)` to change the `k`-th state.
"""
struct ParallelEvolver{IIP, D, T, S<:AbstractVector{T}, F, P, k}
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    states::Vector{S}
    # used only when IIP = true
    dummy::Vector{T}
end

function ParallelEvolver(prob::DiscreteProblem{IIP, D, T, S, F, P}, states) where
    {IIP, D, T, S<:AbstractVector{T}, F, P}
    k = length(states)
    if IIP == true
        s = [Vector(a) for a in states]
    else
        s = [SVector{D, T}(a) for a in states]
    end
    return ParallelEvolver{IIP, D, T, S, F, P, k}(prob, s, Vector(deepcopy(states[1])))
end

"""
    set_state!(pe::ParallelEvolver, x, k)
Set the `k`-th state of the [`ParallelEvolver`](@ref) to `x`.
"""
set_state!(pe::ParallelEvolver{true}, x, k) = (pe.states[k] .= x)
set_state!(pe::ParallelEvolver{false}, x, k) = (pe.states[k] = x)

function evolve!(pe::ParallelEvolver{true, D, T, S, F, P, k},
    N::Int = 1) where {D, T, S<:AbstractVector{T}, F, P, k}
    for j in 1:N
        for i in 1:k
            pe.dummy .= pe.states[i]
            pe.prob.f(pe.states[i], pe.dummy, pe.prob.p)
        end
    end
    return
end

function evolve!(pe::ParallelEvolver{false, D, T, S, F, P, k},
    N::Int = 1) where {D, T, S<:AbstractVector{T}, F, P, k}
    for j in 1:N
        for i in 1:k
            pe.states[i] = pe.prob.f(pe.states[i], pe.prob.p)
        end
    end
    return
end


#####################################################################################
#                              Jacobian Generation                                  #
#####################################################################################
# Here f must be of the form: f(x) -> SVector (ONE ARGUMENT!)
function generate_jacobian_oop(f::F, x::X) where {F, X<:AbstractVector}
    # Test f structure:
    @assert !isinplace(f, 2)
    # Setup config
    cfg = ForwardDiff.JacobianConfig(f, x)
    FDjac(x, p) = ForwardDiff.jacobian(f, x, cfg)
    return FDjac
end

# Here f must be of the form: f(x) -> Number (ONE ARGUMENT!)
function generate_jacobian_oop(f::F, x::X) where {F, X<:Number}
    # Test f structure:
    @assert !isinplace(f, 2)
    FDder(x, p) = ForwardDiff.derivative(f, x)
    return FDder
end


# Here f! must be of the form: f!(dx, x), in-place with 2 arguments!
# It also applies `f!` to `dum` at each call
function generate_jacobian_iip(f!::F, x::X, dum) where {F, X}
    # Test f structure:
    @assert isinplace(f!, 2)
    # Setup config
    cfg = ForwardDiff.JacobianConfig(f!, dum, x)
    # Notice that this version is inefficient: The result of applying f! is
    # already written in `dum` when the Jacobian is calculated. But this is
    # also done during normal evolution, making `f!` being applied twice.
    FDjac!(J, x, p) = ForwardDiff.jacobian!(J, f!, dum, x, cfg)
    return FDjac!
end


#####################################################################################
#                               Parallel Evolvers                                   #
#####################################################################################
"""
    TangentEvolver
A structure that evolves in parallel a discrete system as well as
deviation vectors `ws`, which obey the tangent space dynamics.
Also contains the Jacobian function and matrix of the equations of motion (EOM).

Pass a `TangentEvolver` into the [`DiscreteDynamicalSystem`](@ref) constructor
to use the features of **DynamicalSystems.jl**.

## Description
If ``J`` is the jacobian of the system then the equations for the system
and a deviation vector (or matrix) ``w`` are:
```math
\\begin{aligned}
u_{n+1} &= f(u_n) \\\\
w_{n+1} &= J(u_n) \\times w_n
\\end{aligned}
```
with ``f`` being the EOM function and ``u`` the system state.

Use
```julia
ws = evolve!(ws, tangentevolver, N)
```
to evolve both ``u`` as well as ``w`` for `N` steps. It is necessary
to use the syntax `ws = ...` to be able to evolve any number of ``w``
(which are columns of the matrix `ws`)
for both the in-place and out-of-place versions. (it is not actually necessary to
write `ws = ` for the in-place version)

For out-of-place version `ws` should be an `SMatrix`, wheras for the out-of-place
version it only has to be `<:AbstractMatrix.`

## Constructor
```julia
TangentEvolver(dp::DiscreteProblem [, jacobian [, J]])
```
The `jacobian` is a *function* and (if given) must also be of the same form as the EOM,
`jacobian(x, p) -> SMatrix` for the out-of-place version and
`jacobian!(xnew, x, p)` for the in-place version.
`J` is an initialized Jacobian matrix (only useful in the in-place case).

If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).

The state ``u`` is stored in the field `state` and is
independent of the state of the underlying [`DiscreteProblem`](@ref).

## Resetting - **important**
For the in-place version only, the `TangentEvolver` must be "reset"
to a given `k::Int` in order to evolve a different amount of `ws`. By default
the amount of `ws` (which are stored as a matrix) coincide with the dimension of
the system `D`, but any number `k ≤ D` can be evolved (all in parallel).

Use `reform!(tangentevolver, k::Int, reset_state = true)` to reform the evolver.
By default the function also resets the state to the state of the `DiscreteProblem`.

## Related Functions
Use [`set_state!`](@ref) to change state between steps.
Use [`orthonormal`](@ref) to obtain
a matrix of orthonormal vectors, which can be used as initial conditions for
`evolve!(ws, tangentevolver, N)`.

See [`tangent_integrator`](@ref) for the case of continuous systems.
"""
mutable struct TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M}
    # Utilizes `dummy` field of `prob`
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    jacobian::JA
    J::M
    state::S
    dummyws::Matrix{T}
    # The Type-parameter IAD simply states whether the
    # the Jacobian is autodifferentiated, which only matters
    # in the iip case
end

"""
    reform!(tangentevolver [, k::Int = D] [, reset_state::Bool = true])
Reform the `tangentevolver` so that it can evolve `k` deviation vectors. Only
necessary for in-place form.
"""
function reform!(te::TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M},
    k::Int, reset_state::Bool = true) where {IIP, IAD, D, T, S, F, P, JA, M}
    if reset_state
        set_state!(te, te.prob.u0)
    end
    te.dummyws = zeros(T, D, k)
    return
end

function reform!(te::TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M},
    reset_state::Bool = true) where {IIP, IAD, D, T, S, F, P, JA, M}
    reform!(te, D, reset_state)
end

function get_J(prob::DiscreteProblem, jacob)
    D = dimension(prob)
    if isinplace(prob)
        J = similar(prob.u0, (D,D))
        jacob(J, prob.u0, prob.p)
    else
        J = jacob(prob.u0, prob.p)
    end
    return J
end

# Constructor with Jacobian
function TangentEvolver(prob::DiscreteProblem{IIP, D, T, S, F, P},
    jacobian::JA, J = get_J(prob, jacobian)) where
    {IIP, D, T, S, F, P, JA}
    M = typeof(J)
    return TangentEvolver{IIP, false, D, T, S, F, P, JA, M}(prob,
    jacobian, J, deepcopy(state(prob)), orthonormal(D, D))
end

# Constructor WITHOUT jacobian
function TangentEvolver(prob::DiscreteProblem{IIP, D, T, S, F, P}) where
    {IIP, D, T, S, F, P}
    if !IIP
        reducedeom = (x) -> prob.f(x, prob.p)
        jacob = generate_jacobian_oop(reducedeom, state(prob))
    else
        reducedeom = (dx, x) -> prob.f(dx, x, prob.p)
        # This line ensures that the Jacobian call also applies f in dummy!
        jacob = generate_jacobian_iip(reducedeom, state(prob), prob.dummy)
    end
    JA = typeof(jacob)
    J = get_J(prob, jacob)
    M = typeof(J)
    return TangentEvolver{IIP, true, D, T, S, F, P, JA, M}(prob,
    jacob, J, deepcopy(state(prob)), orthonormal(D, D))
end

state(te::TangentEvolver) = te.state


"""
    set_state!(te::TangentEvolver, x)
Set the state of the [`TangentEvolver`](@ref) to `x`.
"""
set_state!(pe::TangentEvolver{true}, x) = (pe.state .= x)
set_state!(pe::TangentEvolver{false}, x) = (pe.state = x)

"""
    evolve!(ws, te::TangentEvolver, N = 1) -> ws_next
Evolve the `te.state` and and the tangent vectors `ws` for `N` steps.

The function **must** be called as `ws = evolve!(ws, te, N)` in the out-of-place
version. See [`reform!`](@ref) to evolve different amount of deviation vectors.
"""
function evolve!(ws, te::TangentEvolver{true, false}, N::Int = 1)
    # iip with user jacobian
    for j in 1:N
        # println("N = $j")
        # println("te.ds.J = $(te.ds.J)")
        te.prob.dummy .= te.state
        te.jacobian(te.J, te.state, te.prob.p)
        te.prob.f(te.state, te.prob.dummy, te.prob.p)
        te.dummyws .= ws
        A_mul_B!(ws, te.J, te.dummyws)
        # println("after jacobian")
        # println("te.ds.J = $(te.ds.J)")
    end
    return ws
end

# iip with autodiff jacobian:
function evolve!(ws, te::TangentEvolver{true, true}, N::Int = 1)
    for j in 1:N
        # This line applies `f` to `te.prob.dummy`
        te.jacobian(te.J, te.state, te.prob.p)
        te.dummyws .= ws
        A_mul_B!(ws, te.J, te.dummyws)
        # This utilizes the fact that `te.prob.dummy` is evolved:
        te.state .= te.prob.dummy
    end
    return ws
end

# oop version:
function evolve!(ws, te::TangentEvolver{false}, N::Int = 1)
    for j in 1:N
        J = te.jacobian(te.state, te.prob.p)
        te.state = te.prob.f(te.state, te.prob.p)
        ws = J*ws
    end
    return ws
end


#####################################################################################
#                                      DDS                                          #
#####################################################################################
"""
    DiscreteDynamicalSystem <: DynamicalSystem
A structure describing a discrete dynamical system. `DDS` is an alias
to `DiscreteDynamicalSystem`.


## Constructor
```julia
DDS(state, eom, [, jacobian [, J]]; p = nothing)
```
where `eom` is the equations of motion function and `p` is the parameter container,
a *keyword argument*.

The are two "versions" for the `eom`, either in-place (iip) or out-of-place (oop).
Here is how to define them:

* **iip** : The EOM **must** be in the form `eom(x, p) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector` containing the next state.
* **oop** : The EOM **must** be in the form `eom!(xnew, x, p)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

iip is suggested for big systems, whereas oop is suggested for small systems.

The `jacobian` is a *function* and (if given) must also be of the same form as the EOM,
`jacobian(x, p) -> SMatrix` for the out-of-place version and
`jacobian!(xnew, x, p)` for the in-place version.
`J` is an initialized Jacobian matrix (only useful in the in-place case).

If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).

## Fields

* `prob::DiscreteProblem` : The [`DiscreteProblem`](@ref),
  which describes the equations of motion (EOM) and the state of the system.
* `tangent::TangentEvolver` : Dynamics for the tangent space of the system,
  in the form of a [`TangentEvolver`](@ref), which also include the Jacobian
  function.

The low-level constructor simply does `DDS(prob::DiscreteProblem, te::TangentEvolver)`.

## Related Functions
[`state`](@ref), [`set_state!`](@ref),
[`set_parameter!`](@ref), [`jacobian`](@ref),
[`ParallelEvolver`](@ref), [`TangentEvolver`](@ref).
"""
struct DiscreteDynamicalSystem{IIP, IAD, D, T, S, F, P, JA, M} <: DynamicalSystem
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    tangent::TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M}
end
# Alias
DDS = DiscreteDynamicalSystem

# With jacobian
function DiscreteDynamicalSystem(s, eom::F, jacob::JA; p = nothing) where {F, JA}
    prob = DiscreteProblem(s, eom, p)
    J = get_J(prob, jacob)
    tangent = TangentEvolver(prob, jacob, J)
    return DDS(prob, tangent)
end

function DiscreteDynamicalSystem(s, eom::F, jacob::JA, J; p = nothing) where {F, JA}
    prob = DiscreteProblem(s, eom, p)
    tangent = TangentEvolver(prob, jacob, J)
    return DDS(prob, tangent)
end

# Without jacobian
function DiscreteDynamicalSystem(s::S, eom::F; p = nothing) where {S, F}
    prob = DiscreteProblem(s, eom, p)
    tangent = TangentEvolver(prob)
    return DDS(prob, tangent)
end

# Expand methods
for f in (:isinplace, :dimension, :eltype, :statetype, :state, :set_parameter)
    @eval begin
        @inline ($f)(ds::DiscreteDynamicalSystem) = $(f)(ds.prob)
    end
end
set_parameter!(ds::DDS, args...) = set_parameter!(ds.prob, args...)
ParallelEvolver(ds::DDS, states) = ParallelEvolver(ds.prob, states)
set_state!(ds::DDS, xnew) = set_state!(ds.prob, xnew)

# evolve
evolve!(ds::DDS, N::Int = 1) = evolve!(ds.prob, N)
evolve!(u, ds::DDS, N::Int = 1) = evolve!(u, ds.prob, N)
evolve(ds::DDS, N::Int = 1) = evolve(ds.prob, N)
evolve(ds::DDS, N::Int, u) = evolve(ds.prob, N, u)
evolve(ds::DDS, u) = evolve(ds.prob, u)
trajectory(ds::DDS, args...) = trajectory(ds.prob, args...)

"""
    jacobian([J, ] ds::DynamicalSystem, u = state(ds))
Return the Jacobian of the equations of motion at `u`, optionally writting the
result in-place in `J`.
"""
function jacobian(ds::DDS{true}, u = state(ds))
    ds.tangent.jacobian(ds.tangent.J, u, ds.prob.p)
    return ds.tangent.J
end

jacobian(ds::DDS{false}, u = state(ds)) = ds.tangent.jacobian(u, ds.prob.p)

function jacobian(J, ds::DDS, u = state(ds))
    ds.tangent.jacobian(J, u, ds.prob.p)
end

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DDS) =
"$(dimension(ds))-dimensional discrete dynamical system"
Base.summary(ds::DiscreteProblem) =
"$(dimension(ds))-dimensional discrete dynamical problem"
Base.summary(pe::ParallelEvolver) =
"$(dimension(pe.prob))-dimensional discrete parallel evolver"
Base.summary(te::TangentEvolver) =
"$(dimension(te.prob))-dimensional discrete tangent evolver"

function Base.show(io::IO, ds::DDS)
    text = summary(ds)
    print(io, text*"\n",
    " state: $(state(ds))\n", " e.o.m.: $(ds.prob.f)\n",
    " jacobian: $(ds.tangent.jacobian)\n")
end
function Base.show(io::IO, ds::DiscreteProblem)
    text = summary(ds)
    print(io, text*"\n",
    " state: $(state(ds))\n", " e.o.m.: $(ds.f)\n")
end
function Base.show(io::IO, pe::ParallelEvolver)
    text = summary(pe)
    print(io, text*"\n",
    " states: $(tuple(pe.states...))\n", " e.o.m.: $(pe.prob.f)\n")
end
function Base.show(io::IO, te::TangentEvolver)
    text = summary(te)
    print(io, text*"\n",
    " state: $(te.state)\n", " e.o.m.: $(te.prob.f)\n",
    " jacobian: $(te.jacobian)\n")
end
