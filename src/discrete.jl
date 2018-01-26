using StaticArrays, ForwardDiff, Requires

export DiscreteDS, DiscreteDS1D, evolve, trajectory, dimension, state, jacobian
export BigDiscreteDS, DiscreteDynamicalSystem, evolve!
export set_state!

#####################################################################################
#                                   Constructors                                    #
#####################################################################################
"Abstract type representing discrete systems."
abstract type DiscreteDynamicalSystem <: DynamicalSystem end
"""
    DiscreteDS <: DynamicalSystem
`D`-dimensional discrete dynamical system.
## Fields
* `state::SVector{D}` : Current state-vector of the system, stored in the data format
  of `StaticArray`'s `SVector`.
* `eom` (function) : The function that represents the system's equations of motion
  (also called vector field). It **must** be in the form `eom(x, p) -> SVector`
  which means that given a state `x::SVector` and some parameter container
  `p` it returns an `SVector` containing the next state.
* `jacob` (function) : A function that calculates the system's jacobian matrix.
  It **must be** in the form `jacob(x, p) -> SMatrix` which means that given a state
  `x::Svector` and a parameter container `p` it returns an `SMatrix`
  containing the Jacobian at that state.
* `p` : Some kind of container of (initial) parameters. Highly suggested to use
  a subtype of `Array` or [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl).

It is not necessary that `p` is used inside the functions (e.g.
a model without parameters), however
the functions **must be** declared in this format.

Use [`set_state!`](@ref) to change the system's state.

## Constructor
```julia
DiscreteDS(state, eom [, jacob]; parameters = nothing)
```
If the `jacob` is not provided by the user, it is created automatically
using the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).
Notice that if your model has parameters, you *must* give them via the keyword
argument `parameters`.

*Automatic differentiation and parameter changes
works only if the container `p` is changed in-place!*
"""
mutable struct DiscreteDS{D, T<:Number, F, J, P} <: DiscreteDynamicalSystem
    state::SVector{D,T}
    eom::F
    jacob::J
    p::P
end
function DiscreteDS(u0::AbstractVector, eom, jac; parameters = nothing)
    D = length(u0)
    su0 = SVector{D}(u0)
    T = eltype(su0); F = typeof(eom); J = typeof(jac)
    P = typeof(parameters)
    return DiscreteDS{D, T, F, J, P}(su0, eom, jac, parameters)
end
# constructor without jacobian (uses ForwardDiff)
function DiscreteDS(u0::AbstractVector, eom; parameters = nothing)
    su0 = SVector{length(u0)}(u0)
    reducedeom(x) = eom(x, parameters)
    cfg = ForwardDiff.JacobianConfig(eom, u0)
    @inline ForwardDiff_jac(x) = ForwardDiff.jacobian(eom, x, cfg)
    return DiscreteDS(su0, eom, ForwardDiff_jac, parameters)
end

"""
    set_state!(ds::DynamicalSystem, newstate)
Set the state of the system to `newstate`.
"""
set_state!(ds::DiscreteDS, unew) = (ds.state = unew)

"""
    dimension(ds::DynamicalSystem) -> D
Return the dimension of the system.
"""
dimension(::DiscreteDS{D, T, F, J, P}) where {D, T, F, J, P} = D

jacobian(ds::DiscreteDS) = ds.jacob(state(ds), ds.p)

"""
    state(ds::DynamicalSystem) -> u
Return the state of the system.
"""
state(ds::DynamicalSystem) = ds.state

"""
    DiscreteDS1D(state, eom [, deriv]; parameters = nothing) <: DynamicalSystem
One-dimensional discrete dynamical system.
## Fields
* `state::Number` : Current state of the system.
* `eom` (function) : The function that represents the system's equations of motion.
  It **must** be in the form `eom(x, p) -> Number`.
* `deriv` (function) : A function that calculates the system's derivative given
  a state. It **must** be in the form `deriv(x, p) -> Number`.
  If it is not provided by the user
  it is created automatically using the module
  [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).
* `p` : Some kind of container of (initial) parameters. Highly suggested to use
  a subtype of `Array` or [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl).

*Automatic differentiation and parameter changes
works only if the container `p` is changed in-place!*
"""
mutable struct DiscreteDS1D{S<:Number, F, D, P} <: DiscreteDynamicalSystem
    state::S
    eom::F
    deriv::D
    p::P
end
function DiscreteDS1D(x0, eom; parameters = nothing)
    reducedeom(x) = eom(x, p)
    ForwardDiff_der(x) = ForwardDiff.derivative(reducedeom, x)
    DiscreteDS1D(x0, eom, ForwardDiff_der, parameters)
end
DiscreteDS1D(a,b,c;parameters = nothing) = DiscreteDS1D(a,b,c,parameters)

dimension(::DiscreteDS1D) = 1


"""
    BigDiscreteDS <: DynamicalSystem
`D`-dimensional discrete dynamical system (used for big `D`). The equations
for this system perform all operations *in-place*.
## Fields:
* `state::Vector{T}` : Current state-vector of the system.
  Do `state(ds) .= u` to change the state.
* `eom!` (function) : The function that represents the system's equations of motion
  (also called vector field). The function is of the format: `eom!(xnew, x, p)`
  which means that given a state `Vector` `x` and some parameter container `p`,
  it writes in-place the new state in `xnew`.
* `jacob!` (function) : A function that calculates the system's jacobian matrix,
  based on the format: `jacob!(J, x, p)` which means that given a state `Vector`
  `x` it writes in-place the Jacobian in `J`.
* `J::Matrix{T}` : Initialized Jacobian matrix (optional).
* `p` : Some kind of container of (initial) parameters. Highly suggested to use
  a subtype of `Array` or [`LMArray`](https://github.com/JuliaDiffEq/LabelledArrays.jl).
* `dummystate::Vector{T}` : Dummy vector, which most of the time fills the
  role of the previous state in e.g. [`evolve`](@ref).

It is not necessary that `p` is used inside the functions (e.g.
a model without parameters), however
the functions **must be** declared in this format.

Use [`set_state!`](@ref) to change the system's state.

## Constructor
```julia
BigDiscreteDS(state, eom! [, jacob! [, J]]; parameters = nothing)
```

If the `jacob` is not provided by the user, it is created automatically
using the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).
Notice that if your model has parameters, you *must* give them via the keyword
argument `parameters`.

*Automatic differentiation and parameter changes
works only if the container `p` is changed in-place!*
"""
struct BigDiscreteDS{T<:Number, F, JJ, P} <: DiscreteDynamicalSystem
    state::Vector{T}
    eom!::F
    jacob!::JJ
    J::Matrix{T}
    p::P
    dummystate::Vector{T}
end
function BigDiscreteDS(u0, f!, j!,
    J::Matrix = zeros(eltype(u0), length(u0), length(u0));
    parameters = nothing)
    dum = copy(u0)
    j!(J, u0, parameters)
    BigDiscreteDS(u0, f!, j!, J, parameters, dum)
end

# Constructor without jacobian
function BigDiscreteDS(u0, f!; parameters = nothing)
    J = zeros(eltype(u0), length(u0), length(u0))
    dum = copy(u0)

    reducedeom! = (xnew, x) -> f!(xnew, x, parameters)
    cfg = ForwardDiff.JacobianConfig(reducedeom!, dum, u0)
    FD_jacob!(J, x, p) = ForwardDiff.jacobian!(J, reducedeom!, dum, x, cfg)
    FD_jacob!(J, u0, parameters)
    return BigDiscreteDS(u0, f!, FD_jacob!, J, parameters, dum)
end

dimension(ds::BigDiscreteDS) = length(state(ds))

set_state!(ds::BigDiscreteDS, u0) = (ds.state .= u0)

"""
    jacobian(ds::DynamicalSystem) -> J
Return the Jacobian matrix of the equations of motion at the system's current
state and parameters.
"""
jacobian(ds::DynamicalSystem) = (ds.jacob!(ds.J, state(ds), ds.p); ds.J)

#####################################################################################
#                               System Evolution                                    #
#####################################################################################
"""
    evolve(ds::DynamicalSystem, T [, u0]; diff_eq_kwargs = Dict())
Evolve the `state(ds)` (or `u0` if given) for total time `T` and return the
`final_state`. For discrete systems `T` corresponds to steps and
thus it must be integer. For continuous systems `T` can also be a tuple (for `tspan`).

`evolve` *does not store* any information about intermediate steps.
Use [`trajectory`](@ref) if you want to produce a trajectory of the system.
"""
function evolve(ds::DiscreteDynamicalSystem, N::Int, st = state(ds))
    for i in 1:N
        st = ds.eom(st, ds.p)
    end
    return st
end

function evolve(ds::BigDiscreteDS, N::Int, st = deepcopy(state(ds)))
    for i in 1:N
        ds.dummystate .= st
        ds.eom!(st, ds.dummystate, ds.p)
    end
    return st
end

"""
    evolve!(ds::DynamicalSystem, T; diff_eq_kwargs = Dict())
Same as [`evolve`](@ref) but updates the system's state (in-place) with the
final state.

Notice that for continuous systems `ds.prob.u0` is a *reference* to a vector.
Modifying it modifies all other references to this vector, including the state
of other `ContinuousDS` that share the same reference.
"""
evolve!(ds::DiscreteDynamicalSystem, N::Int) = (ds.state = evolve(ds, N))
evolve!(ds::BigDiscreteDS, N::Int) = (ds.state .= evolve(ds, N))


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
function trajectory(ds::DiscreteDS, N::Int)
    st = state(ds)
    ts = Vector{typeof(st)}(N)
    ts[1] = st
    f = ds.eom
    for i in 2:N
        st = f(st, ds.p)
        ts[i] = st
    end
    return Dataset(ts)
end

function trajectory(ds::DiscreteDS1D, N::Int)
    x = deepcopy(state(ds))
    f = ds.eom
    ts = Vector{typeof(x)}(N)
    ts[1] = x
    for i in 2:N
        x = f(x, ds.p)
        ts[i] = x
    end
    return ts
end

function trajectory(ds::BigDiscreteDS, N::Int)
    x = deepcopy(state(ds))
    SV = SVector{dimension(ds), eltype(x)}
    f! = ds.eom!
    ts = Vector{SV}(N)
    ts[1] = SV(x)
    for i in 2:N
        ds.dummystate .= x
        f!(x, ds.dummystate, ds.p)
        ts[i] = SV(x)
    end
    return Dataset(ts)
end





#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::DiscreteDS) = "$(dimension(ds))-dimensional discrete dynamical system"
Base.summary(ds::BigDiscreteDS) = "$(dimension(ds))-dimensional Big discrete system"
Base.summary(ds::DiscreteDS1D) = "1-dimensional discrete dynamical system"

function Base.show(io::IO, ds::DynamicalSystem)
    text = summary(ds)
    eom = typeof(ds) <: BigDiscreteDS ? ds.eom! : ds.eom
    print(io, text*"\n",
    " state: $(state(ds))\n", " e.o.m.: $(eom)\n")
end

@require Juno begin
    function Juno.render(i::Juno.Inline, ds::DynamicalSystem)
        t = Juno.render(i, Juno.defaultrepr(ds))
        text = summary(ds)
        t[:head] = Juno.render(i, Text(text))
        t[:children] = t[:children][1:2] # remove showing field dummystate
        t
    end
end
