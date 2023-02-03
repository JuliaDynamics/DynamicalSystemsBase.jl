export TangentDynamicalSystem, XXXX

# Implementation: the state and deviation vectors are combined in a matrix.
# For oop this is SMatrix, for IIP it is the same type as `J0`.
# The standard analytic systems with matrix state are used. No fancy
# dedicated discrete tangent integrator anymore. The amount of deviation vectors
# become a type parameter for efficient static matrix computations.

##################################################################################
# Type definition and docs
##################################################################################

"""
    TangentDynamicalSystem(ds::AnalyticRuleSystem; J = nothing, k = dimension(ds))

A dynamical system that bundles the evolution of `ds`
(which must be an [`AnalyticRuleSystem`](@ref)) and `k` deviation vectors
that are evolved according to the _dynamics in the tangent space_
(also called linearized dynamics or the tangent dynamics).

The state of `ds` **must** be an `AbstractVector` for `TangentDynamicalSystem`.

## Keyword arguments

- `k` or `Q0`: If `k::Int` is given, the first `k` columns of the identity matrix are used
  as deviation vectors. Otherwise `Q0` can be given which is a matrix with each column a
  deviation vector. It must hold that `size(Q, 1) == dimension(ds)`.
  You can use [`orthonormal`](@ref) for random orthonormal vectors.
- `J` and `J0`: See section "Jacobian" below.

## Description

Let ``u`` be the state of `ds`, and ``y`` a deviation (or perturbation) vector.
These two are evolved in parallel according to

```math
\\begin{array}{rcl}
\\dot{\\vec{x}}&=& f(\\vec{x}) \\\\
\\dot{Y} &=& J_f(\\vec{x}) \\cdot Y
\\end{array}
\\quad \\text{or}\\quad
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
struct TangentDynamicalSystem{D, J} <: ParallelDynamicalSystem
    ds::D      # standard dynamical system but with rule the tangent space
    original_f # no type parameterization here, this field is only for printing
    J::J
    isautodiff::Bool
end

# it is practically identical to `ParallelDynamicalSystemAnalytic`

##################################################################################
# Initialization
##################################################################################
function TangentDynamicalSystem(ds::AnalyticRuleSystem{IIP};
        J = nothing, k::Int = dimension(ds), Q0 = diagm(ones(dimension(ds)))[:, 1:k],
        J0 = zeros(dimension(ds), dimension(ds)),
    ) where {IIP}
    current_state(ds) isa AbstractVector || error("State of `ds` must be vector-like.")
    D = dimension(ds)
    size(Q0, 1) ≠ D && throw(ArgumentError("size(Q, 1) ≠ dimension(ds)"))
    size(Q0, 2) > D && throw(ArgumentError("size(Q, 2) > dimension(ds)"))
    size(J0) ≠ (D, D) && throw(ArgumentError("size(J0) ≠ (dimension(ds), dimension(ds))"))
    Jf = jacobianf(ds, J)
    newrule = tangent_rule(f, J, ...)
    J0_correct = correct_matrix_type(Val{IIP}(), Q0)
    tands = ...
    return TangentDynamicalSystem(tands, dynamic_rule(ds))
end

# Make jacobian function
jacobianf(ds, J) = J
jacobianf(ds::AnalyticRuleSystem{IIP}, ::Nothing) where {IIP} =
autodiff_jacobian(
    dynamical_rule(ds), Val{IIP}(), current_state(ds),
    current_parameters(ds), current_time(ds)
)
# in place
function autodiff_jacobian(@nospecialize(f::F), ::Val{true}, s, p, t) where {F}
    dum = deepcopy(s)
    inplace_f_2args = (y, x) -> f(y, x, p, t)
    cfg = ForwardDiff.JacobianConfig(inplace_f_2args, dum, s)
    jac! = (J, u, p, t) -> ForwardDiff.jacobian!(
        J, inplace_f_2args, dum, u, cfg, Val{false}()
    )
    return jac!
end
# out of place
function autodiff_jacobian(@nospecialize(f::F), ::Val{false}, args...) where {F}
    # SVector methods do *not* use the config and hence don't care about `args`
    return (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
end


function correct_matrix_type(::Val{false}, Q::AbstractMatrix)
    A, B = size(Q)
    SMatrix{A, B}(Q)
end
correct_matrix_type(::Val{false}, Q::SMatrix) = Q
correct_matrix_type(::Val{true}, Q::AbstractMatrix) = ismutable(Q) ? Q : Array(Q)

##################################################################################
# Create tangent rule
##################################################################################

# OOP Tangent Space dynamics
function tangent_rule(f::F, J::JAC, J0, ::Val{false}, ::Val{k}) where {F, JAC, k}
    # Initial matrix `J0` is ignored
    ws_index = SVector{k, Int}(2:(k+1)...)
    tangentf = TangentOOP{F, JAC, k}(f, J, ws_index)
    return tangentf
end
struct TangentOOP{F, JAC, k} <: Function
    f::F
    J::JAC
    ws::SVector{k, Int}
end
function (tan::TangentOOP)(u, p, t)
    @inbounds s = u[:, 1]
    du = tan.f(s, p, t)
    J = tan.J(s, p, t)
    @inbounds dW = J*u[:, tan.ws]
    return hcat(du, dW)
end
