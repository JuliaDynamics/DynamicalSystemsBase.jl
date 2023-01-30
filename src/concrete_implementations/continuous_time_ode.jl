"""

## DifferentialEquations.jl keyword arguments
Continuous dynamical systems are evolved via the solvers of DifferentialEquations.jl.
Functions in DynamicalSystems.jl allow providing options to these solvers via the
`diffeq` keyword. For example you could use `diffeq = (abstol = 1e-9, reltol = 1e-9)`.
If you want to specify a solver, do so by using the keyword `alg`, e.g.:
`diffeq = (alg = Tsit5(), maxiters = 100000)`. This requires you to have been first
`using OrdinaryDiffEq` to access the solvers. See the
`CDS_KWARGS` variable for the default values we use.
Notice that `diffeq` keywords can also include `callback` for [event handling](http://docs.juliadiffeq.org/latest/features/callback_functions.html).

Keep in mind that the default solver is `SimpleATsit5`, which only supports
adaptive time-stepping. Use `(alg = SimpleTsit5(), dt = your_step_size)` as keywords
for a non-adaptive time stepping solver.
"""