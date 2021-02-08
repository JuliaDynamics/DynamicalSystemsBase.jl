
#######################################################################################
#                        Tangent Dynamics (aka linearized dynamics)                   #
#######################################################################################

# IIP Tangent Space dynamics
function create_tangent(@nospecialize(f::F), @nospecialize(jacobian::JAC), J::JM,
    ::Val{true}, ::Val{k}) where {F, JAC, JM, k}
    J = deepcopy(J)
    tangentf = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        f(view(du, :, 1), uv, p, t)
        jacobian(J, uv, p, t)
        mul!((@view du[:, 2:(k+1)]), J, (@view u[:, 2:(k+1)]))
        nothing
    end
    return tangentf
end
# for the case of autodiffed systems, a specialized version is created
# so that f! is not called twice in ForwardDiff
function create_tangent_iad(f::F, J::JM, u, p, t, ::Val{k}) where {F, JM, k}
    let
        J = deepcopy(J)
        cfg = ForwardDiff.JacobianConfig(
            (du, u) -> f(du, u, p, p), deepcopy(u), deepcopy(u)
        )
        tangentf = (du, u, p, t) -> begin
            uv = @view u[:, 1]
            ForwardDiff.jacobian!(
                J, (du, u) -> f(du, u, p, t), view(du, :, 1), uv, cfg, Val{false}()
            )
            mul!((@view du[:, 2:k+1]), J, (@view u[:, 2:k+1]))
            nothing
        end
        return tangentf
    end
end


# OOP Tangent Space dynamics
function create_tangent(f::F, jacobian::JAC, J::JM,
    ::Val{false}, ::Val{k}) where {F, JAC, JM, k}

    ws_index = SVector{k, Int}(2:(k+1)...)
    tangentf = TangentOOP{F, JAC, k}(f, jacobian, ws_index)
    return tangentf
end
struct TangentOOP{F, JAC, k} <: Function
    f::F
    jacobian::JAC
    ws::SVector{k, Int}
end
function (tan::TangentOOP)(u, p, t)
    @inbounds s = u[:, 1]
    du = tan.f(s, p, t)
    J = tan.jacobian(s, p, t)
    @inbounds dW = J*u[:, tan.ws]
    return hcat(du, dW)
end


#######################################################################################
#                                Parallel Dynamics                                    #
#######################################################################################
# Create equations of motion of evolving states in parallel
function create_parallel(ds::DS{true}, states)
    st = [Vector(s) for s in states]
    L = length(st)
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            @inbounds ds.f(du[i], u[i], p, t)
        end
    end
    return paralleleom, st
end

function create_parallel(ds::DS{false}, states)
    D = dimension(ds)
    st = [SVector{D}(s) for s in states]
    L = length(st)
    # The following may be inneficient
    paralleleom = (du, u, p, t) -> begin
        for i in 1:L
            @inbounds du[i] = ds.f(u[i], p, t)
        end
    end
    return paralleleom, st
end
