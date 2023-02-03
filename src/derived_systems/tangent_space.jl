export TangentDynamicalSystem, XXXX

# Implementation: the state and deviation vectors are combined in a matrix.
# For oop this is SMatrix, for IIP it is the same type as `J0`.
# The standard analytic systems with matrix state are used. No fancy
# dedicated discrete tangent integrator anymore. The amount of deviation vectors
# become a type parameter for efficient static matrix computations.

"""
    TangentDynamicalSystem(ds::AnalyticRuleSystem; J = nothing, k = dimension(ds))

A dynamical system that bundles the evolution of `ds`
(which must be an [`AnalyticRuleSystem`](@ref)) and `k` deviation vectors
that are evolved according to the _dynamics in the tangent space_
(also called linearized dynamics or the tangent dynamics).

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
for the out-of-place version or `J(M, u, p, n)` for the in-place version.
in both cases `M` is a matrix whose columns are the deviation vectors.

By default `J = nothing`.  In this case `J` is constructed automatically using
the module [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/).
Even though `ForwardDiff` is very fast, depending on your exact system you might
gain significant speed-up by providing a hand-coded Jacobian and so it is recommended.

The keyword `J0` allows you to pass an initialized Jacobian matrix `J0`.
This is useful for large in-place systems where only a few components of the Jacobian change
during the time evolution. `J0` can be a sparse or any other matrix type.
If not given, `zeros(dimension(ds), k)` is used.
"""
function XXXX end