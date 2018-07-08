using BenchmarkTools, DynamicalSystemsBase

const SUITE = BenchmarkGroup(["DynamicalSystemsBase"])

include("reconstructions.jl")
include("integrators.jl")
