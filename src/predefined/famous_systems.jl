"""
Submodule of the module `DynamicalSystemsBase`, which contains pre-defined
famous systems. This is provided purely as a convenience.
Nothing here is tested, nor guaranteed to be stable in future versions.

Predefined systems exist in the `Systems` submodule in the form of functions that
return a `DynamicalSystem`. They are accessed like:
```julia
ds = Systems.lorenz(u0; ρ = 32.0)
```

For some systems, a Jacobian function is also defined.
The naming convention for the jacobian function is `\$(name)_jacob`.
So, for the above example we have `J = Systems.lorenz_jacob`.
"""
module Systems
using DynamicalSystemsBase

using DynamicalSystemsBase: DDS
include("discrete_famous_systems.jl")

using DynamicalSystemsBase: CDS
include("continuous_famous_systems.jl")

end# Systems module
