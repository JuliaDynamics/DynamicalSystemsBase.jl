export TangentDynamicalSystem, current_deviations, set_deviations!
using LinearAlgebra: mul!, diagm

# Implementation: the state and deviation vectors are combined in a matrix.
# First column is formal state, all remaining columns are deviation vectors.
# For oop this is SMatrix, for IIP it is the same type as `Q0`.
# The standard analytic systems with matrix state are used. No fancy
# dedicated discrete tangent integrator anymore. The amount of deviation vectors
# become a type parameter for efficient static matrix computations.

###########################################################################################
# Type definition and docs
###########################################################################################

"""
    TangentDynamicalSystem <: DynamicalSystem
    TangentDynamicalSystem(ds::CoreDynamicalSystem; kwargs...)

A dynamical system that bundles the evolution of `ds`
(which must be an [`CoreDynamicalSystem`](@ref)) and `k` deviation vectors
that are evolved according to the _dynamics in the tangent space_
(also called linearized dynamics or the tangent dynamics).

The state of `ds` **must** be an `AbstractVector` for `TangentDynamicalSystem`.

`TangentDynamicalSystem` follows the [`DynamicalSystem`](@ref)
interface with the following adjustments:

- `reinit!` takes an additional keyword `Q0` (with same default as below)
- The additional functions [`current_deviations`](@ref) and
  [`set_deviations!`](@ref) are provided for the deviation vectors.

## Keyword arguments

- `k` or `Q0`: `Q0` represents the initial deviation vectors (each column = 1 vector).
  If `k::Int` is given, a matrix `Q0` is created with the first `k` columns of
  the identity matrix. Otherwise `Q0` can be given directly as a matrix.
  It must hold that `size(Q, 1) == dimension(ds)`.
  You can use [`orthonormal`](@ref) for random orthonormal vectors.
  By default `k = dimension(ds)` is used.
- `u0 = current_state(ds)`: Starting state.
- `J` and `J0`: See section "Jacobian" below.

## Description

Let ``u`` be the state of `ds`, and ``y`` a deviation (or perturbation) vector.
These two are evolved in parallel according to

```math
\\begin{array}{rcl}
\\frac{d\\vec{x}}{dt} &=& f(\\vec{x}) \\\\
\\frac{dY}{dt} &=& J_f(\\vec{x}) \\cdot Y
\\end{array}
\\quad \\mathrm{or}\\quad
\\begin{array}{rcl}
\\vec{x}_{n+1} &=& f(\\vec{x}_n) \\\\
Y_{n+1} &=& J_f(\\vec{x}_n) \\cdot Y_n.
\\end{array}
```
for continuous or discrete time respectively. Here ``f`` is the [`dynamic_rule`](@ref)`(ds)`
and ``J_f`` is the Jacobian of ``f``.

## Jacobian

The keyword `J` provides the Jacobian function. It must be a Julia function
in the same form as `f`, the [`dynamic_rule`](@ref).
Specifically, `J(u, p, n) -> M::SMatrix`
for the out-of-place version or `J(M, u, p, n)` for the in-place version
acting in-place on `M`.
in both cases `M` is a matrix whose columns are the deviation vectors.

By default `J = nothing`.  In this case `J` is constructed automatically using
the module [`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl),
hence its limitations also apply here.
Even though `ForwardDiff` is very fast, depending on your exact system you might
gain significant speed-up by providing a hand-coded Jacobian and so it is recommended.
Additionally, automatic and in-place Jacobians cannot be time dependent.

The keyword `J0` allows you to pass an initialized Jacobian matrix `J0`.
This is useful for large in-place systems where only a few components of the Jacobian change
during the time evolution. `J0` can be a sparse or any other matrix type.
If not given, a matrix of zeros is used. `J0` is ignored for out of place systems.
"""
struct TangentDynamicalSystem{IIP, D} <: ParallelDynamicalSystem
    ds::D      # standard dynamical system but with rule the tangent space
    # no type parameterization here, this field is only for printing
    original_f
    J
end

additional_details(tands::TangentDynamicalSystem) = [
    "jacobian" => isnothing(tands.J) ? "ForwardDiff" : rulestring(tands.J),
    "deviation vectors" => size(current_deviations(tands), 2),
]

# it is practically identical to `TangentDynamicalSystem`

function TangentDynamicalSystem(ds::CoreDynamicalSystem{IIP};
        J = nothing, k::Int = dimension(ds), Q0 = default_deviations(dimension(ds), k),
        J0 = zeros(dimension(ds), dimension(ds)), u0 = current_state(ds),
    ) where {IIP}

    # Valid input checks:
    current_state(ds) isa AbstractVector || error("State of `ds` must be vector-like.")
    D = dimension(ds)
    k = size(Q0, 2)
    f = dynamic_rule(ds)
    size(Q0, 1) ≠ D && throw(ArgumentError("size(Q, 1) ≠ dimension(ds)"))
    size(Q0, 2) > D && throw(ArgumentError("size(Q, 2) > dimension(ds)"))
    size(J0) ≠ (D, D) && throw(ArgumentError("size(J0) ≠ (dimension(ds), dimension(ds))"))

    # Create jacobian, tangent rule, initial state
    u0_correct = correct_state(Val{IIP}(), u0)
    Q0_correct = correct_matrix_type(Val{IIP}(), Q0)
    newstate = hcat(u0_correct, Q0_correct)
    newrule = tangent_rule(f, J, J0, Val{IIP}(), Val{k}(), u0_correct)

    # Pass everything to analytic system constructors
    cp = current_parameters(ds)
    if ds isa DeterministicIteratedMap
        tands = DeterministicIteratedMap(newrule, newstate, cp)
    elseif ds isa CoupledODEs
        T = eltype(newstate)
        prob = ODEProblem{IIP}(newrule, newstate, (T(initial_time(ds)), T(Inf)), cp)
        tands = CoupledODEs(prob, ds.diffeq; internalnorm = matrixnorm)
    end
    return TangentDynamicalSystem{IIP, typeof(tands)}(tands, f, J)
end

function correct_matrix_type(::Val{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
correct_matrix_type(::Val{false}, Q::SMatrix) = Q
correct_matrix_type(::Val{true}, Q::AbstractMatrix) = ismutable(Q) ? Q : Array(Q)

###########################################################################################
# Creation of tangent rule
###########################################################################################
import ForwardDiff
# IIP Tangent space dynamics
function tangent_rule(f::F, J::JAC, J0, ::Val{true}, ::Val{k}, u0) where {F, JAC, k}
    tangentf = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        f(view(du, :, 1), uv, p, t)
        J(J0, uv, p, t)
        mul!((@view du[:, 2:(k+1)]), J0, (@view u[:, 2:(k+1)]))
        nothing
    end
    return tangentf
end
# for the case of autodiffed systems, a specialized version is created
# so that f! is not called twice in ForwardDiff
function tangent_rule(f::F, ::Nothing, J0, ::Val{true}, ::Val{k}, u0) where {F, k}
    let
        cfg = ForwardDiff.JacobianConfig(
            (du, u) -> f(du, u, p, p), deepcopy(u0), deepcopy(u0)
        )
        tangentf = (du, u, p, t) -> begin
            uv = @view u[:, 1]
            ForwardDiff.jacobian!(
                J0, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
            )
            mul!((@view du[:, 2:k+1]), J0, (@view u[:, 2:k+1]))
            nothing
        end
        return tangentf
    end
end

# OOP Tangent space dynamics
function tangent_rule(f::F, J::JAC, J0, ::Val{false}, ::Val{k}, u0) where {F, JAC, k}
    # out of place
    if JAC == Nothing
        # There is no config needed here
        Jf = (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
    else
        Jf = J
    end
    # Initial matrix `J0` is ignored
    ws_index = SVector{k, Int}(2:(k+1)...)
    tangentf = TangentOOP(f, Jf, ws_index)
    return tangentf
end
struct TangentOOP{F, JAC, k} <: Function
    f::F
    J::JAC
    ws::SVector{k, Int}
end
function (tan::TangentOOP)(u, p, t)
    # @show u
    @inbounds s = u[:, 1]
    du = tan.f(s, p, t)
    J = tan.J(s, p, t)
    @inbounds dW = J*u[:, tan.ws]
    return hcat(du, dW)
end

###########################################################################################
# Extensions
###########################################################################################
dynamic_rule(tands::TangentDynamicalSystem) = tands.original_f
(tands::TangentDynamicalSystem)(t::Real) = tands.ds(t)[:, 1]

for f in (:current_time, :initial_time, :isdiscretetime,
        :current_parameters, :initial_parameters, :isinplace,
    )
    @eval $(f)(tands::TangentDynamicalSystem, args...; kw...) = $(f)(tands.ds, args...; kw...)
end
current_state(t::TangentDynamicalSystem{true}) = view(current_state(t.ds), :, 1)
current_state(t::TangentDynamicalSystem{false}) = current_state(t.ds)[:, 1]
initial_state(t::TangentDynamicalSystem{true}) = view(initial_state(t.ds), :, 1)
initial_state(t::TangentDynamicalSystem{false}) = initial_state(t.ds)[:, 1]

"""
    current_deviations(tands::TangentDynamicalSystem)

Return the deviation vectors of `tands` as a matrix with each column a vector.
"""
current_deviations(t::TangentDynamicalSystem{true}) = @view(current_state(t.ds)[:, 2:end])
function current_deviations(t::TangentDynamicalSystem{false})
    # state is an SMatrix
    U = current_state(t.ds)
    return U[:, dynamic_rule(t.ds).ws] # from TangentOOP
end

# Dedicated step so that it retunrs the system itself
SciMLBase.step!(tands::TangentDynamicalSystem, args...) = (step!(tands.ds, args...); tands)

function set_state!(t::TangentDynamicalSystem{true}, u)
    current_state(t) .= u
    set_state!(t.ds, current_state(t.ds))
end
function set_state!(t::TangentDynamicalSystem{false}, u)
    u_correct = typeof(current_state(t))(u)
    U = hcat(u_correct, current_deviations(t))
    set_state!(t.ds, U)
end

"""
    set_deviations!(tands::TangentDynamicalSystem, Q)

Set the deviation vectors of `tands` to be `Q`, a matrix with each column a vector.
"""
function set_deviations!(t::TangentDynamicalSystem{true}, Q)
    current_deviations(t) .= Q
    set_state!(t.ds, current_state(t.ds))
end
function set_deviations!(t::TangentDynamicalSystem{false}, Q)
    Q_correct = typeof(current_deviations(t))(Q)
    U = hcat(current_state(t), Q_correct)
    set_state!(t.ds, U)
end

function SciMLBase.reinit!(tands::TangentDynamicalSystem{IIP}, u = initial_state(tands);
        p = current_parameters(tands), t0 = initial_time(tands), Q0 = default_deviations(tands)
    ) where {IIP}
    isnothing(u) && return
    u_correct = correct_state(Val{IIP}(), u)
    Q0_correct = correct_matrix_type(Val{IIP}(), Q0)
    if IIP
        current_state(tands) .= u_correct
        current_deviations(tands) .= Q0_correct
        U = current_state(tands.ds)
    else
        U = hcat(u_correct, Q0_correct)
    end
    reinit!(tands.ds, U; p, t0)
end

function default_deviations(tands)
    k = size(current_deviations(tands), 2)
    return default_deviations(dimension(tands), k)
end
default_deviations(D::Int, k::Int) = diagm(ones(D))[:, 1:k]