# v3.0

Complete rewrite of the package. The DynamicalSystems.jl v3 changelog summarizes the highlights. Here we will try to list all changes, but it will be difficult to provide a changelog given that everything was changed.

## Enhancements
- `DynamicalSystem` now defines a proper, well-thought-out interface that is implemented from all its concrete subtypes. See its docstring for details.
- All `DynamicalSystem` implementations are now "integrators": mutable objects that are stepped with `step!`. They need to be deepcopied for parallelizing via threads.
- `ParallelDynamicalSystem` now works for any kind of dynamical system.
- `trajectory` works for any conceivable thing that extends the `DynamicalSystem` interface

## Deprecations
- `DiscreteDynamicalSystem -> DeterministicIteratedMap`
- `ContinuousDynamicalSystem -> CoupledODEs`
- `parallel_integrator -> ParallelDynamicalSystem`
- `tangent_integrator -> TangentDynamicalSystem`
- `stroboscopicmap -> StroboscopicMap`
- `poincaremap -> PoincareMap`

## Majorly Breaking
- The `Discrete/ContinuousDynamicalSystem` constructors no longer accept a Jacobian. Use the dedicated `TangentDynamicalSystem` for something that represents the tangent space and can be given to downstream functions such as `lyapunovspectrum`. As a result, none of the predefined systems come with a hand coded Jacobian. The function is still available for manual use nevertheless.
- The keyword `diffeq` does not exist anymore and is not given to any downstream functions such as `lyapunovspectrum`. The only struct that cares about DifferentialEquations.jl arguments is `CoupledODEs` so it is the only one that accepts `diffeq` keyword.
- `trajectory` now returns the actual trajectory _and_ the time vector: `X, t = trajectory(ds, ...)`

# v2.9
- All code related to the poincare map integrator have been moved here from ChaosTools.jl.

# v2.8
- New functions `get_parameters` and `get_parameter` that simplify interacting with the parameters of a dynamical system (and drastically simplify source code of `set_parameter!`).

# v2.7.2
- Integer Δt is now enforced in trajectory calculation of discrete systems. Not doing so led to silently incorrect behavior that the user wouldn't expect.
- Fixed `set_parameter!` not working for `projected_integrator`.

# v2.7
* New function `get_states` that returns an iterator over states contained in a `parallel_integrator`.
# v2.6
* Added new `GeneralizedDynamicalSystem` abstract type that is an umbrella term.
# v2.5
* Added new `current_time(integ)` function.
* Added `trajectory` for `projected_integrator` and `stroboscopicmap`.
* Source code of `trajectory` has been expended and generalized. It should now work out of the box for any object that defines the integrator API, as defined in the DynamicalSystems.jl documentation.
* Added new boolean function `isdiscretetime` that works for all systems/integrators.
# v2.4
* Added new type of integrator: `stroboscopicmap`.
* Added new type of integrator: `projected_integrator`.
# v2.3
* Added several new famous systems.
# v2.2
* `get_state(parallel_integrator, k)` now returns a view in case the integrator is in-place.
# v2.1.1
* Fix `set_state!` not working for in-place parallel integrator
# v2.1.0
* Now `set_parameter!` can also work with parameter containers that are composite types.
* Added some new pre-defined systems used in synchronization studies, like the Kuramoto model or coupled Roessler oscillators or nonlinearly coupled logistic maps.
* The integrator `SimpleTsit5` is also exported. This integrator is non-adaptive, while the default integrator of the library `SimpleATsit5` is adaptive.

# v2.0.0
* The keyword `dt` of `trajectory` has been renamed to `Δt`.
  This keyword had conflicts with the options of DifferentialEquations.jl.
  No warning can be thrown for this change, and users still using `dt` will
  have it silently propagated as keyword to the diffeq solvers.


# v1.8.11
* More performant version of `set_deviations!(integ, Q)` when `Q` is the result of
  a QR-decomposition.
# v1.8.3
* Added `grebogi_map` dynamical system.
# v1.8
* `trajectory` now supports keyword `save_idxs`.
# v1.7
* New system: Thomas cyclically symmetric
# v1.6.1
* Arbitrary keywords can be propagated into `trajectory` for discrete systems
  (which are unused, this is simply for unifying syntax with continuous systems)
# v1.6.0
* Bugfix of `ODEProblem(ContinuousDynamicalSystem)` where the parameters of the system where not passed properly.
* Move to Julia 1.5.
* Allow `ODEProblem(cds)` to take a different state optionally

# v1.5.4
* Critical bugfix/mentioning of how 1-D systems work.
# v1.5
* Added function that provides initial conditions for henon-heiles system at given energy, `Systems.henonheiles_ics(E, n)`
* More famous systems and a lot more Jacobians to existing systems (#93)
# v1.4
* added Pomaeu-Manneville map into the famous systems.
# v1.3
* The specialized integrators (tangent & parallel) now implement an internal norm that only evaluates norm of the main state, instead of using the other parallel states or deviation vectors. (#86)
* Bunch of bugfixes and performance improvements (see git history)

# v1.2
In version `1.2` we have moved to `SciMLBase 5.0+`. In addition to that we have changed the default integrator to `SimpleATsit5` from module `SimpleDiffEq`. This is not a breaking change and you can use any of the previous integrators. **It must be stated though that numeric results you obtained using the default integrator will now be slightly different**.

In addition, be sure that you have version `SimpleDiffEq 0.3.0` or greater (which is what `DynamicalSystemsBase 1.2.3` guarantees).

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
