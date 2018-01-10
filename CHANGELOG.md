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
