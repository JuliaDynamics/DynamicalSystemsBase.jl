using DynamicalSystemsBase

ti = time()

# Mathematics:
include("math_tests.jl")
# Discrete:
include("discrete_types.jl")
include("discrete_systems.jl")
include("discrete_tangent.jl")
# Continuous:
include("continuous_systems.jl")
# Dataset:
include("dataset_tests.jl")
# Reconstruction:
include("reconstruction_tests.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, 3), " seconds or ", round(ti/60, 3), " minutes")
