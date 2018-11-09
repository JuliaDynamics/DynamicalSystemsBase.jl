using DynamicalSystemsBase

ti = time()

# Systems:
include("dynsys_types.jl")
include("dynsys_tangent.jl")
include("dynsys_inference.jl")
include("continuous_systems.jl")
include("discrete_systems.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, digits=3), " seconds or ", round(ti/60, digits=3), " minutes")
