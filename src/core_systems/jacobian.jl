export jacobian

import ForwardDiff

"""
    jacobian(ds::CoreDynamicalSystem)

Construct the Jacobian rule for the dynamical system `ds`.
This is done via automatic differentiation using module 
[`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl).

## Description

For out-of-place systems, `jacobian` returns the Jacobian rule as a 
function `Jf(u, p, t) -> J0::SMatrix`. Calling `Jf(u, p, t)` will compute the Jacobian 
at the state `u`, parameters `p` and time `t` and return the result as `J0`.
For in-place systems, `jacobian` returns the Jacobian rule as a function 
`Jf!(J0, u, p, t)`. Calling `Jf!(J0, u, p, t)` will compute the Jacobian 
at the state `u`, parameters `p` and time `t` and save the result in `J0`.
"""
function jacobian(ds::CoreDynamicalSystem{IIP}) where {IIP}
    _jacobian(ds, Val{IIP}())
end

function _jacobian(ds, ::Val{true})
    f = dynamic_rule(ds)
    u0 = current_state(ds)
    cfg = ForwardDiff.JacobianConfig(
        (du, u) -> f(du, u, p, p), deepcopy(u0), deepcopy(u0)
    )
    Jf! = (J0, u, p, t) -> begin
        uv = @view u[:, 1]
        du = copy(u)
        ForwardDiff.jacobian!(
            J0, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
        )
        nothing
    end
    return Jf!
end

function _jacobian(ds, ::Val{false})
    f = dynamic_rule(ds)
    Jf = (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
    return Jf
end