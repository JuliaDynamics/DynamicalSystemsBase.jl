using DynamicalSystemsBase

ti = time()

# Mathematics:
include("math_tests.jl")
# System Evolution:
include("discrete_systems.jl")
include("continuous_systems.jl")

ti = time() - ti
println("\nTest took total time of:")
println(round(ti, 3), " seconds or ", round(ti/60, 3), " minutes")
