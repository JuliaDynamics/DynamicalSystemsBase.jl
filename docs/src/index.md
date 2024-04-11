# DynamicalSystemsBase.jl

```@docs
DynamicalSystemsBase
```

!!! note "Tutorial and examples at DynamicalSystems.jl docs!
    Please visit the documentation of the main DynamicalSystems.jl docs for a tutorial and examples on using the interface.

## The `DynamicalSystem` API

```@docs
DynamicalSystem
```

```@docs
current_state
initial_state
observe_state
state_name
current_parameters
current_parameter
parameter_name
initial_parameters
isdeterministic
isdiscretetime
dynamic_rule
current_time
initial_time
isinplace(::DynamicalSystem)
successful_step
referrenced_sciml_model
```

```@docs
reinit!(::DynamicalSystem, ::AbstractDict; kwargs...)
set_state!
set_parameter!
set_parameters!
```

## Time evolution
```@docs
step!(::DynamicalSystem, args...; kwargs...)
trajectory
StateSpaceSet
```

## `DeterministicIteratedMap`
```@docs
DeterministicIteratedMap
```

## `CoupledODEs`
```@docs
CoupledODEs
```

## `StroboscopicMap`
```@docs
StroboscopicMap
```

## `PoincareMap`
```@docs
PoincareMap
current_crossing_time
poincaresos
```

## `TangentDynamicalSystem`
```@docs
CoreDynamicalSystem
TangentDynamicalSystem
current_deviations
set_deviations!
orthonormal
```

## `ProjectedDynamicalSystem`
```@docs
ProjectedDynamicalSystem
```

## `ParallelDynamicalSystem`
```@docs
ParallelDynamicalSystem
initial_states
current_states
```

## `ArbitrarySteppable`
```@docs
ArbitrarySteppable
```

## Parallelization

Since `DynamicalSystem`s are mutable, one needs to copy them before parallelizing, to avoid having to deal with complicated race conditions etc. The simplest way is with `deepcopy`. Here is an example block that shows how to parallelize calling some expensive function (e.g., calculating the Lyapunov exponent) over a parameter range using `Threads`:

```julia
ds = DynamicalSystem(f, u, p) # some concrete implementation
parameters = 0:0.01:1
outputs = zeros(length(parameters))

# Since `DynamicalSystem`s are mutable, we need to copy to parallelize
systems = [deepcopy(ds) for _ in 1:Threads.nthreads()-1]
pushfirst!(systems, ds) # we can save 1 copy

Threads.@threads for i in eachindex(parameters)
    system = systems[Threads.threadid()]
    set_parameter!(system, 1, parameters[i])
    outputs[i] = expensive_function(system, args...)
end
```

## Advanced example

This is an advanced example of making an in-place implementation of coupled [standard maps](https://en.wikipedia.org/wiki/Standard_map). It will utilize a handcoded Jacobian, a sparse matrix for the Jacobinan, a default initial Jacobian matrix, as well as function-like-objects as the dynamic rule.

Coupled standard maps is a deterministic iterated map that can have arbitrary number of equations of motion, since you can couple `N` standard maps which are 2D maps, like so:

```math
\theta_{i}' = \theta_i + p_{i}' \\
p_{i}' = p_i + k_i\sin(\theta_i) - \Gamma \left[\sin(\theta_{i+1} - \theta_{i}) + \sin(\theta_{i-1} - \theta_{i}) \right]
```

To model this, we will make a dedicated `struct`, which is parameterized on the
number of coupled maps:

```@example MAIN
using DynamicalSystemsBase

struct CoupledStandardMaps{N}
    idxs::SVector{N, Int}
    idxsm1::SVector{N, Int}
    idxsp1::SVector{N, Int}
end
```

(what these fields are will become apparent later)

We initialize the struct with the amount of standard maps we want to couple,
and we also define appropriate parameters:

```@example MAIN
M = 5  # couple number
u0 = 0.001rand(2M) #initial state
ks = 0.9ones(M) # nonlinearity parameters
Γ = 1.0 # coupling strength
p = (ks, Γ) # parameter container

# Create struct:
SV = SVector{M, Int}
idxs = SV(1:M...) # indexes of thetas
idxsm1 = SV(circshift(idxs, +1)...)  #indexes of thetas - 1
idxsp1 = SV(circshift(idxs, -1)...)  #indexes of thetas + 1
# So that:
# x[i] ≡ θᵢ
# x[[idxsp1[i]]] ≡ θᵢ+₁
# x[[idxsm1[i]]] ≡ θᵢ-₁
csm = CoupledStandardMaps{M}(idxs, idxsm1, idxsp1)
```

We will now use this struct to define a [function-like-object](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects), a Type that also acts as a function

```@example MAIN
function (f::CoupledStandardMaps{N})(xnew::AbstractVector, x, p, n) where {N}
    ks, Γ = p
    @inbounds for i in f.idxs

        xnew[i+N] = mod2pi(
            x[i+N] + ks[i]*sin(x[i]) -
            Γ*(sin(x[f.idxsp1[i]] - x[i]) + sin(x[f.idxsm1[i]] - x[i]))
        )

        xnew[i] = mod2pi(x[i] + xnew[i+N])
    end
    return nothing
end
```

We will use *the same* `struct` to create a function for the Jacobian:

```@example MAIN
function (f::CoupledStandardMaps{M})(
    J::AbstractMatrix, x, p, n) where {M}

    ks, Γ = p
    # x[i] ≡ θᵢ
    # x[[idxsp1[i]]] ≡ θᵢ+₁
    # x[[idxsm1[i]]] ≡ θᵢ-₁
    @inbounds for i in f.idxs
        cosθ = cos(x[i])
        cosθp= cos(x[f.idxsp1[i]] - x[i])
        cosθm= cos(x[f.idxsm1[i]] - x[i])
        J[i+M, i] = ks[i]*cosθ + Γ*(cosθp + cosθm)
        J[i+M, f.idxsm1[i]] = - Γ*cosθm
        J[i+M, f.idxsp1[i]] = - Γ*cosθp
        J[i, i] = 1 + J[i+M, i]
        J[i, f.idxsm1[i]] = J[i+M, f.idxsm1[i]]
        J[i, f.idxsp1[i]] = J[i+M, f.idxsp1[i]]
    end
    return nothing
end
```

This is possible because the system state is a `Vector` while the Jacobian is a `Matrix`, so multiple dispatch can differentiate between the two.

Notice in addition, that the Jacobian function accesses *only half the elements of the matrix*. This is intentional, and takes advantage of the fact that the
other half is constant. We can leverage this further, by making the Jacobian a sparse matrix. Because the `DynamicalSystem` constructors allow us to give in a pre-initialized Jacobian matrix, we take advantage of that and create:
```@example MAIN
using SparseArrays
J = zeros(eltype(u0), 2M, 2M)
# Set ∂/∂p entries (they are eye(M,M))
# And they dont change they are constants
for i in idxs
    J[i, i+M] = 1
    J[i+M, i+M] = 1
end
sparseJ = sparse(J)

csm(sparseJ, u0, p, 0) # apply Jacobian to initial state
sparseJ
```

Now we are ready to create our dynamical system

```@example MAIN
ds = DeterministicIteratedMap(csm, u0, p)
```

Of course, the reason we went through all this trouble was to make a [`TangentDynamicalSystem`](@ref), that can actually use the Jacobian function.

```@example MAIN
tands = TangentDynamicalSystem(ds; J = csm, J0 = sparseJ, k = M)
```

```@example MAIN
step!(tands, 5)
current_deviations(tands)
```

(the deviation vectors will increase in magnitude rapidly because the dynamical system is chaotic)
