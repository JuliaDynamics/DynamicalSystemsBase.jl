infomsg = """
~~~ DynamicalSystemsBase.jl update message ~~~

We have changed the fundamental way users define equations of motion and Jacobians
for all subtypes of `DynamicalSystem`! In short, parameters are now directly passed
into the equations of motion function, e.g. `f(du, u, p, t)`. These updates are in-line
with DifferentialEquations v4.0 release.

Please see the documentation strings of the system types you are using!

Also, be sure to visit the updated documentation here:
https://juliadynamics.github.io/DynamicalSystems.jl/latest/

"""

info(infomsg)
