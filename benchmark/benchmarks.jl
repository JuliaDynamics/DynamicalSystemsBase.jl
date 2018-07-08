using BenchmarkTools, DynamicalSystemsBase

const SUITE = BenchmarkGroup(["DynamicalSystemsBase"])

include("integrators.jl")
