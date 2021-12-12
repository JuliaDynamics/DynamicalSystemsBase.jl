DIFFEQ_DEP_WARN = """
Direct propagation of keyword arguments to DifferentialEquations.jl is deprecated.
From now on pass any DiffEq-related keywords as a `NamedTuple` using the
explicit keyword `diffeq` instead.
"""