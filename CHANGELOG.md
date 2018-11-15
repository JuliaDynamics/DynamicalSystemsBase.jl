Changelog of `DynamicalSystemsBase`.

# master

# v1.1

* All functionality related to `neighborhood`, `reconstruct` and `Dataset` has moved to a new package: [`DelayEmbeddings`](https://github.com/JuliaDynamics/DelayEmbeddings.jl). It is reexported by `DynamicalSystemsBase`, which is now (as the name suggests) the basis package for defining `DynamicalSystem`.

# v1.0

* First major release (long term stability).

# v0.12.0
## New Features
* `orthonormal` is 100x more performant.
* Super duper cool new printing for `DynamicalSystem`.
* Added tests etc. and now we are sure we can support Sparse matrix jacobians.
* Testing of specialized integrators with DiffEqCallbacks.
* Parameters are printed in the `DynamicalSystem`.

# 0.11.0

## Breaking
* Dropped support of Julia versions < 0.7
* `Reconstruction` does not exist anymore. Instead, `reconstruct` is used in it's
  place. The function now also always returns a `Dataset`.
* In the `reconstruct` function, `D` now stands for the number of temporal neighbors
  which is **one less** than the dimensionality of the reconstructed space.
    *  This change was done because it is more intuitive and more scalable when considering multiple timeseries or spatio temporal timeseries.


* Re-worked the internals of `DynamicalSystem` (although the API remained the same): Now `DynamicalSystem` only stores what is absolutely necessary and creates directly integrators when need be. A "problem" is not stored anymore.
This also lead to re-working of how keyword arguments are handled. I am
very happy to say that these changes reduced *tremendously* the source code!

## New Features
* Exported a 3-argument `reinit!(integ, u0::AbstractVector, Q0::AbstractMatrix)` that takes `Q0` as the third argument
  and reinits safely any integrator.

* `reconstruct` creates internally a subtype of `AbstractEmbedding`. These objects can be used as functors to create the `i`-th reconstructed vector on demand. This also improved the source code clarity significantly.

* `tangent_` and `parallel_integrator` can now accept callbacks for continuous systems. In general you could pass to the constructors any keyword acceptable by `init` of DiffEq.

# v0.10

*some of the following changes are breaking*

* `state` function no longer exists and has been merged into `get_state`.
* Created specialized tangent integrator for discrete systems, which is about
  20% faster. This is a "breaking" change.
* Created functions `get_state`, `get_deviations`, `set_state!` and
  `set_deviations!` that always correctly return either the system
  state or the deviation vectors (in a form of a matrix) from *any* integrator!!!
* Dynamical Systems with matrices as states are no longer available.
* Added note in reconstruction about how many temporal neighbors there are.

# v0.9
* Theiler window is now part of the `neighborhood` function
* Methods that estimate parameters for `Reconstruction` have moved back to
  `ChaosTools`.

# v0.8
## New features
* Multi-dimensional Reconstruction
* Multi-time Reconstruction
* Multi-dimensional, multi-time Reconstruction
* Methods for estimating Reconstruction parameters are now here.
* Reconstruction from Dataset
## Breaking
* Corrected name `helies -> heiles`
* Reconstructions now have field `delay` which is the delay time/times and are
  instead parameterized on the type of the delay.

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
* Added energy conservation option to the Henon Heiles system
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
