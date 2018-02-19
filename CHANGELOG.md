# v0.6
## Massively Breaking
* Complete overhaul of all of DynamicalSystems. More details
  are in the official documentation (because I would had to write 10 pages of changelog
  otherwise). The constructors also changed signature.
* All `DynamicalSystem`s are now  *immutable* and contain a problem,
  a jacobian and a jacobian matrix.
* Advantage of Multiple dispatch is taken to distinguish between in-place,
  out of place, continuous, discrete, heaven, hell, whatever.
* All internals use `integrator` and `step!`.
* `variational_integrator` renamed to `tangent_integrator`.

## Non-breaking
* Massive performance boost of up to 8x in system evolution of continuous systems.
* increased the interaction between DiffEq and DynamicalSystems.
* A dedicated integrator for discrete systems that is super fast is now employed
  by default. It follows the same high level DiffEq interface but the implementation
  is totally internal.
* Out-of-place continuous systems are allowed. All famous systems are actually
  out of place now.
* `parallel_integrator` added.

# v0.5.0

## Massively Breaking
* All system type definitions have changed. See the new documentation strings
  or the new docs! We are now having the syntax `eom!(du, u, p, t)` and the
  parameters are passed directly into the function!!!

# Non-breaking
* Improved the algorithm that converts a Dataset to a Matrix. It is now not only
  faster, but also more clear!

# v0.4.1
* (Breaking?) Moved `estimate_delay` to ChaosTools.jl
* Added `gissinger` system.
* Added `columns` function for columns of dataset.

# v0.4.0
## Breaking
* By default now all `ContinuousDynamicalSystems` are solved using `Vern9` as a
  solver and tolerances of `1e-9`, both abstol and reltol.
## Non-breaking
* Added energy conservation option to the Henon Helies system
* All `ContinuousDS` evolution now internally passes thrgouh the `get_sol` function,
  which improves the clarity and stability of the ecosystem greatly!!!
* Improved stability in propagating `solve` keywords.
* `get_sol` now returns solution and time vector for generality purposes.
* `get_sol` is now also exported.
* Internal `ODEProblem` constructor correctly merges different callbacks.

# v0.3.3
## Non-breaking
* Bugfix of `eltype` of `Reconstruction`.
* Added `circlemap` to `Systems`.
* Bugfix on `Dataset` that incorrect methods were being called due to missing `<:`.
* Method `Base.getindex(d::AbstractDataset{D,T}, ::Colon, j<:AbstractVector{Int}) ` now exists!
* Added function `evolve!`.
* Clearly state that we do not support matrices in the equations of motion.
* Bugfixes regarding ODEProblem.

# v0.3.2
## Non-breaking
* Orders of magnitude speed-up in conversions between `Matrix` and `Dataset`,
  because now methods use `transpose` internally and only `reinterpret`.

# v0.3.1
* Added `jacobian` function
* Removed `EomVector` nonsense.
* Now `trajectory` correctly gives equi-spaced points when the ODEProblem has
"special" callbacks.

# v0.3.0
## BREAKING
* The type `ContinuousDS` has been completely overhauled.
  * Now the type contains an `ODEProblem` instead of `state` and `eom!`.
  * Now the form of equations of motion is expected as `(t, u, du)`.
  * We are able to handle non-autonomous systems with "simple" time dependence.
  * This also allows to use callbacks natively tied to the system! This means
    that you can create the simple form of Hybrid systems!

# v0.2.1
## Non-breaking
* Added function `minmaxima` that calculates maxima and minima at the same time for
  datasets.

# v0.2.0
## Non-breaking
* Minor docstring improvements and IO improvements.
* Datasets can now be accessed with ranges as well.
* Added the Duffing oscillator in the predefined systems.
* Changed :bug: where `get_solver` would remove the entry.
* Added the possibility to provide a state in *all* evolving functions.
* Added Shinriki oscillator (chaotic 3D flow with period doubling).

# v0.1.0
Initial release, see: https://juliadynamics.github.io/DynamicalSystems.jl/v0.8.0/
for the features up to this point.
