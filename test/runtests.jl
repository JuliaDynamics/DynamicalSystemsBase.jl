using DynamicalSystemsBase

ti = time()

# Systems:
include("dynsys_types.jl")
include("dynsys_tangent.jl")
include("dynsys_inference.jl")
include("continuous_systems.jl")
include("discrete_systems.jl")

# Dataset:
include("dataset_tests.jl")
# Reconstruction:
include("reconstruction_tests.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
