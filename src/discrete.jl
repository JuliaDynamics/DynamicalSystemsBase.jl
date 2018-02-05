using StaticArrays, ForwardDiff
import DiffEqBase: isinplace
import Base: eltype

export DiscreteDynamicalSystem, DDS, DiscreteProblem, DynamicalSystem
export state, jacobian, isinplace, dimension, statetype, state
export set_state!, set_parameter!, TangentEvolver, ParallelEvolver
export set_tangent!
export evolve, evolve!

abstract type DynamicalSystem end

#####################################################################################
#                              Jacobian Generation                                  #
#####################################################################################

# Here f must be of the form: f(x) -> SVector (ONE ARGUMENT!)
function generate_jacobian_oop(f::F, x::X) where {F, X}
    # Test f structure:
    @assert !isinplace(f, 2)
    # Setup config
    cfg = ForwardDiff.JacobianConfig(f, x)
    FDjac(x, p) = ForwardDiff.jacobian(f, x, cfg)
    return FDjac
end

# Here f! must be of the form: f!(dx, x), in-place with 2 arguments!
function generate_jacobian_iip(f!::F, x::X) where {F, X}
    # Test f structure:
    @assert isinplace(f!, 2)
    # Setup config
    dum = deepcopy(x)
    cfg = ForwardDiff.JacobianConfig(f!, dum, x)
    # Notice that this version is inefficient: The result of applying f! is
    # already written in `dum` when the Jacobian is calculated. But this is
    # also done during normal evolution, making `f!` being applied twice.
    FDjac!(J, x, p) = ForwardDiff.jacobian!(J, f!, dum, x, cfg)
    return FDjac!, dum
end


#####################################################################################
#                                  Discrete System                                  #
#####################################################################################
mutable struct DiscreteProblem{IIP, D, T, S<:AbstractVector{T}, F, P}
    u0::S # initial state
    f::F # more similarity with ODEProblem
    p::P
end

function DiscreteProblem(s, eom::F, p::P) where {F, P}
    D = length(s)
    T = eltype(s)
    iip = isinplace(eom, 3)
    iip || (@assert typeof(eom(s, p)) <: SVector)
    iip && (x = deepcopy(s); eom(x, s, p); @assert x!=s)
    u = iip ? Vector(s) : SVector{D}(s)
    S = typeof(u)
    DiscreteProblem{iip, D, T, S, F, P}(u, eom, p)
end

"""
    isinplace(ds::DynamicalSystem) -> Bool
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
    DiscreteDynamicalSystem <: DynamicalSystem
A structure describing a discrete dynamical system. `DDS` is an alias
to `DiscreteDynamicalSystem`.

`DDS` contains the equations of motion (EOM), the initial state and the
the Jacobian of the EOM. The are two "versions" for `DDS`, depending on whether the
EOM are in-place (iip) or out-of-place (oop).

Here is how to define them:

* **iip** : The EOM **must** be in the form `eom(x, p) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector` containing the next state.
* **oop** : The EOM **must** be in the form `eom!(xnew, x, p)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.

iip is suggested for big systems, whereas oop is suggested for small systems.

## Constructor
```julia
DDS(state, eom, p [, jacobian [, J]])
```
where `p` is the parameter container. Pass `nothing` if the dynamical system
does not have parameters, otherwise we highly suggest to use
a subtype of `Array` or [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl).

The `jacobian` is a *function* and (if given) must also be of the same form as the EOM,
`jacobian(x, p) -> SMatrix` for the oop and `jacobian!(xnew, x, p)` for the iip.
`J` is an initialized Jacobian matrix (only useful in the iip case).

The constructor deduces automatically whether the EOM are iip or oop.
If `jacobian` is not given, it is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).

## Related Functions
[`state`](@ref), [`set_state!`](@ref),
[`set_parameter!`](@ref), [`jacobian`](@ref),
[`ParallelEvolver`](@ref), [`TangentEvolver`](@ref).
"""
struct DiscreteDynamicalSystem{IIP, D, T, S, F, P, JA, M} <: DynamicalSystem
    prob::DiscreteProblem{IIP, D, T, S, F, P}
    jacobian::JA
    # The following 2 are used only in the case of IIP = true
    dummy::S
    J::M
    # To solve DynamicalSystemsBase.jl#17
    isautodiff::Bool
end
# Alias
DDS = DiscreteDynamicalSystem

# With jacobian and J
function DiscreteDynamicalSystem(s::S, eom::F, p::P, jacob::JA) where {S, F, P, JA}
    prob = DiscreteProblem(s, eom, p)
    iip = isinplace(prob)
    D = dimension(prob)
    if iip
        J = similar(s, (D,D))
        jacob(J, s, prob.p)
    else
        J = jacob(s, prob.p)
    end
    return DiscreteDynamicalSystem(prob, jacob, deepcopy(s), J, false)
end

# With jacobian but no J
function DiscreteDynamicalSystem(s::S, eom::F, p::P, jacob::JA, J) where {S, F, P, JA}
    prob = DiscreteProblem(s, eom, p)
    iip = isinplace(prob)
    return DiscreteDynamicalSystem(prob, jacob, deepcopy(s), J, false)
end

# Without jacobian
function DiscreteDynamicalSystem(s::S, eom::F, p::P) where {S, F, P}
    prob = DiscreteProblem(s, eom, p)
    iip = isinplace(prob)
    if !iip
        reducedeom = (x) -> eom(x, prob.p)
        jacob = generate_jacobian_oop(reducedeom, s)
        dum = deepcopy(s)
    else
        reducedeom = (dx, x) -> eom(dx, x, prob.p)
        jacob, dum = generate_jacobian_iip(reducedeom, s)
    end
    J = begin
        D = dimension(prob)
        if iip
            J = similar(s, (D,D))
            jacob(J, s, prob.p)
            J
        else
            J = jacob(s, prob.p)
        end
    end

    return DiscreteDynamicalSystem(prob, jacob, dum, J, true)
end

# Expand methods
for f in (:isinplace, :dimension, :eltype, :statetype, :state, :set_parameter)
    @eval begin
        @inline ($f)(ds::DiscreteDynamicalSystem) = $(f)(ds.prob)
    end
end
set_parameter!(ds::DDS, args...) = set_parameter!(ds.prob, args...)

# set_state
"""
    set_state!(ds::DynamicalSystem, unew)
Set the state of the system to `unew`.
"""
function set_state!(ds::DDS, xnew)
    ds.prob.u0 = xnew
end

"""
    jacobian(ds::DynamicalSystem, u = state(ds))
Return the Jacobian of the equations of motion at `u`.
"""
function jacobian(ds::DDS{true}, u = state(ds))
    ds.jacobian(ds.J, u, ds.prob.p)
    return ds.J
end

jacobian(ds::DDS{false}, u = state(ds)) = ds.jacobian(u, ds.prob.p)




#####################################################################################
#                                 Time-Evolution                                    #
#####################################################################################
#= Methods necessary:
evolve(ds [, N] [, u])
evolve!([u], ds, N)
=#
"""
    evolve(ds::DynamicalSystem, [, T] [, u0]; diff_eq_kwargs = Dict())
Evolve `u0` (or `state(ds)` if not given) for total time `T` (or `1` if not given)
and return the `final_state`.

For discrete systems `T` corresponds to steps and
thus it must be integer. For continuous systems `T` can also be a tuple (for `tspan`).

`evolve` *does not store* any information about intermediate steps.
Use [`trajectory`](@ref) if you want to produce a trajectory of the system.
See also `[`trajectory`](@ref)` for using `diff_eq_kwargs`.
"""
evolve(ds::DDS{true}, u = state(ds)) = (ds.prob.f(ds.dummy, u, ds.prob.p); ds.dummy)
evolve(ds::DDS{false}, u = state(ds)) = ds.prob.f(u, ds.prob.p)

function evolve(ds::DDS{true}, N::Int, u = state(ds))
    D = dimension(ds)
    u0 = SVector{D}(u)
    ds.dummy .= u
    for i in 1:N
        ds.prob.f(ds.prob.u0, ds.dummy, ds.prob.p)
        ds.dummy .= u
    end
    uret = SVector{D}(ds.prob.u0)
    ds.prob.u0 .= u0
    return Vector(uret)
end

function evolve(ds::DDS{false}, N::Int, u0 = state(ds))
    for i in 1:N
        u0 = ds.prob.f(u0, ds.prob.p)
    end
    return u0
end

"""
    evolve!(ds::DynamicalSystem, T; diff_eq_kwargs = Dict())
Same as [`evolve`](@ref) but updates the system's state (in-place) with the
final state. See [`trajectory`](@ref) for `diff_eq_kwargs`.

```julia
evolve!(pe::ParallelEvolver, N = 1)
evolve!(te::TangentEvolver, N = 1)
```
Evolve `pe` or `te` for `N` steps in-place.

*Note* - `evolve!` is non-allocating for discrete systems.
"""
evolve!(u, ds::DDS{true}) = (ds.dummy .= u; ds.prob.f(u, ds.dummy, ds.prob.p))
function evolve!(u, ds::DDS{true}, N::Int)
    for i in 1:N
        ds.dummy .= u
        ds.prob.f(u, ds.dummy, ds.prob.p)
    end
    return
end
evolve!(ds::DDS{true}) = evolve!(ds.prob.u0, ds)
evolve!(ds::DDS{true}, N::Int) = evolve!(ds.prob.u0, ds, N)

evolve!(u, ds::DDS{false}) = (u .= ds.prob.f(u, ds.prob.p))
evolve!(u, ds::DDS{false}, N::Int) = (u .= evolve(ds, N, u))
evolve!(ds::DDS{false}, N::Int = 1) = (ds.prob.u0 = evolve(ds, N))


"""
```julia
trajectory(ds::DynamicalSystem, T; kwargs...) -> dataset
```
Return a dataset what will contain the trajectory of the sytem,
after evolving it for time `T`. See [`Dataset`](@ref) for info on how to
manipulate this object.

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
function trajectory(ds::DDS{true}, N::Int, u = state(ds))
    SV = SVector{dimension(ds), eltype(u)}
    f! = ds.prob.f
    ts = Vector{SV}(N)
    ts[1] = SV(u)
    for i in 2:N
        ds.dummy .= ts[i-1]
        f!(ds.prob.u0, ds.dummy, ds.prob.p)
        ts[i] = SV(ds.prob.u0)
    end
    ds.prob.u0 .= ts[1]
    return Dataset(ts)
end

function trajectory(ds::DDS{false}, N::Int, st = state(ds))
    SV = SVector{dimension(ds), eltype(st)}
    ts = Vector{SV}(N)
    ts[1] = st
    f = ds.prob.f
    for i in 2:N
        st = f(st, ds.prob.p)
        ts[i] = st
    end
    return Dataset(ts)
end


#####################################################################################
#                          Parallel/Tangent Evolvers                                #
#####################################################################################
"""
    ParallelEvolver(ds::DiscreteDynamicalSystem, states) -> `pe`
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

ParallelEvolver(ds::DDS, states) = ParallelEvolver(ds.prob, states)

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




#### TAngent evolver
"""
    TangentEvolver(ds::DiscreteDynamicalSystem, ws | k) -> `te`
Return a structure `te` that evolves in parallel the system as well as
deviation vectors `ws` (vector of vectors or
matrix), which obey the tangent space dynamics.
If an integer `k` is given instead of `ws`,
then `k` random orthonormal deviation vectors are chosen.

Use [`evolve!`](@ref)`(te, N)` to evolve (in-place) for `N` steps.

## Description
If ``J`` is the jacobian of the system then the equations for the system
and a deviation vector (or matrix) ``w`` are:
```math
\\begin{aligned}
u_{n+1} &= f(u_n) \\\\
w_{n+1} &= J(u_n) \\cdot w_n
\\end{aligned}
```
with ``f`` being the equations of motion function and ``u`` the system state.

The deviation vectors ``w`` (field `te.ws`) are stored as a matrix, with each
column being a deviation vector.

Use [`set_state!`](@ref) or [`set_tangent!`](@ref) to change
the state or `ws` between steps.

See [`tangent_integrator`](@ref) for the case of continuous systems.
"""
mutable struct TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M}
    # The jacobian ds.J
    ds::DDS{IIP, D, T, S, F, P, JA, M}
    state::S
    ws::M
    dummyws::M
    # The Type-parameter IAD simply states whether there
    # the Jacobian is autodifferentiated, which only matters
    # in the iip case
end

function TangentEvolver(ds::DDS, k::Int)
    D = dimension(ds)
    ws = orthonormal(D, k)
    WS = isinplace(ds) ? ws : to_Smatrix(ws)
    TangentEvolver(ds, WS)
end

function TangentEvolver(ds::DDS{IIP, D, T, S, F, P, JA, M},
    ws::AbstractMatrix) where {IIP, D, T, S, F, P, JA, M}
    IAD = ds.isautodiff
    WS = isinplace(ds) ? ws : to_Smatrix(ws)
    TangentEvolver{IIP, IAD, D, T, S, F, P, JA, M}(
    ds, deepcopy(state(ds)), WS, deepcopy(WS))
end

state(te::TangentEvolver) = te.state


"""
    set_state!(te::TangentEvolver, x)
Set the state of the [`TangentEvolver`](@ref) to `x`.

*Note* - `set_state!` is non-allocating.
"""
set_state!(pe::TangentEvolver{true}, x) = (pe.state .= x)
set_state!(pe::TangentEvolver{false}, x) = (pe.state = x)

"""
    set_tangent!(te::TangentEvolver, ws)
Set the deviation vectors of the [`TangentEvolver`](@ref) to `ws`.
"""
set_tangent!(te::TangentEvolver{true}, x) = (te.ws .= x)
set_tangent!(te::TangentEvolver{false}, x) = (te.ws = x)


function evolve!(te::TangentEvolver{true, false}, N::Int = 1)
    for j in 1:N
        te.ds.dummy .= te.state
        te.ds.prob.f(te.state, te.ds.dummy, te.ds.prob.p)
        te.ds.jacobian(te.ds.J, te.state, te.ds.prob.p)
        te.dummyws .= te.ws
        A_mul_B!(te.ws, te.ds.J, te.dummyws)
    end
    return
end

function evolve!(te::TangentEvolver{true, true}, N::Int = 1)
    te.ds.dummy .= te.state
    for j in 1:N
        # The following line also applues ds.prob.f on ds.dummy
        te.ds.jacobian(te.ds.J, te.state, te.ds.prob.p)
        te.dummyws .= te.ws
        A_mul_B!(te.ws, te.ds.J, te.dummyws)
        te.state .= te.ds.dummy
    end
    return
end

function evolve!(te::TangentEvolver{false}, N::Int = 1)
    for j in 1:N
        te.state = te.ds.prob.f(te.state, te.ds.prob.p)
        J = te.ds.jacobian(te.state, te.ds.prob.p)
        te.ws = J*te.ws
    end
    return
end



#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DDS) =
"$(dimension(ds))-dimensional discrete dynamical system"
Base.summary(pe::ParallelEvolver) =
"$(dimension(pe.prob))-dimensional discrete parallel evolver"
Base.summary(te::TangentEvolver) =
"$(dimension(te.ds))-dimensional discrete tangent evolver"

function Base.show(io::IO, ds::DDS)
    text = summary(ds)
    print(io, text*"\n",
    " state: $(state(ds))\n", " e.o.m.: $(ds.prob.f)\n")
end
function Base.show(io::IO, pe::ParallelEvolver)
    text = summary(pe)
    print(io, text*"\n",
    " states: $(tuple(pe.states...))\n", " e.o.m.: $(pe.prob.f)\n")
end
function Base.show(io::IO, te::TangentEvolver)
    text = summary(te)
    print(io, text*"\n",
    " state: $(te.state)\n", " e.o.m.: $(te.ds.prob.f)\n",
    " ws: $(te.ws)\n")
end
