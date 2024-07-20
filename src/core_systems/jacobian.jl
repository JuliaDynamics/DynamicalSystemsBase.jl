export jacobian

import ForwardDiff

function jacobian(ds::CoreDynamicalSystem{IIP}) where {IIP}
    _jacobian(ds, Val{IIP}())
end

function _jacobian(ds, ::Val{true})
    f = dynamic_rule(ds)
    u0 = current_state(ds)
    cfg = ForwardDiff.JacobianConfig(
        (du, u) -> f(du, u, p, p), deepcopy(u0), deepcopy(u0)
    )
    Jf = (J0, u, p, t) -> begin
        uv = @view u[:, 1]
        du = copy(u)
        ForwardDiff.jacobian!(
            J0, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
        )
        nothing
    end
    return Jf
end

function _jacobian(ds, ::Val{false})
    f = dynamic_rule(ds)
    Jf = (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
    return Jf
end