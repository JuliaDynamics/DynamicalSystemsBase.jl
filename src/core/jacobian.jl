# TODO: Where to put this file?

export jacobian

import ForwardDiff

function jacobian(ds::CoreDynamicalSystem{IIP}, J0=nothing) where {IIP}
    f = dynamic_rule(ds)
    u0 = current_state(ds)
    _jacobian(f, J0, Val{IIP}(), u0)

end

function _jacobian(f, J0, ::Val{true}, u0)
    cfg = ForwardDiff.JacobianConfig(
        (du, u) -> f(du, u, p, p), deepcopy(u0), deepcopy(u0)
    )
    Jf = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        ForwardDiff.jacobian!(
            J0, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
        )
        nothing
    end
    return Jf
end

function _jacobian(f, J0, ::Val{false}, u0)
    # Initial matrix `J0` is ignored
    Jf = (u, p, t) -> ForwardDiff.jacobian((x) -> f(x, p, t), u)
    return Jf
end


function oop(u, p, t)
    return p[1] * SVector(u[1], -u[2])
end

function iip(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = -p[1] * u[2]
    return nothing
end

#%%
oopds=CoupledODEs(oop, [1.0, 1.0], [1.0])
oopjac=jacobian(oopds)
oopjac(current_state(oopds), current_parameters(oopds), 0.0)

iipds=CoupledODEs(iip, [1.0, 1.0], [1.0])
J0 = zeros(2, 2)
iipjac=jacobian(iipds, J0)
du = zeros(2)
iipjac(du, current_state(iipds), current_parameters(iipds), 0.0)
J0

