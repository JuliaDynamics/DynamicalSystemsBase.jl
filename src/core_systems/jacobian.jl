export jacobian

import ForwardDiff

"""
    jacobian(ds::CoreDynamicalSystem)

Construct the Jacobian rule for the dynamical system `ds`.
If the system already has a Jacobian rule constructed via ModelingToolkit.jl it returns this,
otherwise it constructs the Jacobian rule with automatic differentiation using module
[`ForwardDiff`](https://github.com/JuliaDiff/ForwardDiff.jl).

## Description

For out-of-place systems, `jacobian` returns the Jacobian rule as a
function `Jf(u, p, t = 0) -> J0::SMatrix`. Calling `Jf(u, p, t)` will compute the Jacobian
at the state `u`, parameters `p` and time `t` and return the result as `J0`.
For in-place systems, `jacobian` returns the Jacobian rule as a function
`Jf!(J0, u, p, t = 0)`. Calling `Jf!(J0, u, p)` will compute the Jacobian
at the state `u`, parameters `p` and time `t` and save the result in `J0`.
"""
function jacobian(ds::CoreDynamicalSystem{IIP}) where {IIP}
    if ds isa ContinuousTimeDynamicalSystem
        # TODO: This is the correct API way to obtain the Jacobian,
        # however it relies on MTK dependency, so we can't do it.
        # if has_referrenced_model(ds)
        #     model = referrenced_sciml_model(ds)
        #     Joop, Jiip = generate_jacobian(model; expression = Val{false})
        #     if IIP
        #         jac = Jiip
        #     else
        #         jac = Joop
        #     end
        # end
        prob = referrenced_sciml_prob(ds)
        if prob.f isa SciMLBase.AbstractDiffEqFunction && !isnothing(prob.f.jac)
            jac = prob.f.jac
        else
            jac = _jacobian(ds, Val{IIP}())
        end
    else
        jac = _jacobian(ds, Val{IIP}())
    end
    return jac
end

function _jacobian(ds, ::Val{true})
    f = dynamic_rule(ds)
    u0 = current_state(ds)
    p = current_parameters(ds)
    t = initial_time(ds)
    cfg = ForwardDiff.JacobianConfig(
        (du, u) -> f(du, u, p, t), deepcopy(u0), deepcopy(u0)
    )
    Jf! = (J0, u, p, t = 0) -> begin
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
    Jf = (u, p, t = 0) -> ForwardDiff.jacobian(x -> f(x, p, t), u)
    return Jf
end

jacobian(ds::CoupledSDEs) = jacobian(CoupledODEs(ds))
