export tangent_integrator


### Tangent integrators ###

# in-place autodifferentiated jacobian helper struct
struct TangentIIP{F, JM, CFG}
    f::F     # original eom
    J::JM    # Jacobian matrix (written in-place)
    cfg::CFG # Jacobian Config
end
function (j::TangentIIP)(du, u, p, t)
    reducedf = (du, u) -> j.f(du, u, p, t)
    # The following line applies f to du[:,1] and calculates jacobian
    ForwardDiff.jacobian!(j.J, reducedf, view(du, :, 1), view(u, :, 1),
    j.cfg, Val{false}())
    # This performs dY/dt = J(u) ⋅ Y
    A_mul_B!((@view du[:, 2:end]), j.J, (@view u[:, 2:end]))
    return
end
# In-place, autodifferentiated version:
function tangent_integrator(ds::DS{true, true}, Q0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, u0 = ds.prob.u0
    )

    D = dimension(ds)
    initstate = hcat(u0, Q0)

    # Create tangent eom:
    reducedeom = (du, u) -> ds.prob.f(du, u, ds.prob.p, ds.prob.t)
    cfg = ForwardDiff.JacobianConfig(reducedeom, rand(3), rand(3))
    tangenteom = TangentIIP(ds.prob.f, similar(ds.prob.u0, D, D), cfg)

    if problemtype(ds) == ODEProblem
        solver, newkw = extract_solver(diff_eq_kwargs)
        tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, solver; newkw..., save_everystep = false)
    elseif problemtype(ds) == DiscreteProblem
        tanprob = DiscreteProblem(tangenteom, initstate, DDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, FunctionMap(), save_everystep = false)
    end
end



# In-place version:
"""
    tangent_integrator(ds, Q0; u0, diff_eq_kwargs)
"""
function tangent_integrator(ds::DS{true, false}, Q0;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)

    D = dimension(ds)
    initstate = hcat(u0, Q0)
    J = jacobian(ds)

    tangenteom = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        ds.prob.f(view(du, :, 1), uv, p, t)
        ds.jacobian(J, uv, p, t)
        A_mul_B!((@view du[:, 2:end]), J, (@view u[:, 2:end]))
        nothing
    end

    if problemtype(ds) == ODEProblem
        solver, newkw = extract_solver(diff_eq_kwargs)
        tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, solver; newkw..., save_everystep = false)
    elseif problemtype(ds) == DiscreteProblem
        tanprob = DiscreteProblem(tangenteom, initstate, DDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, FunctionMap(), save_everystep = false)
    end
end



# out-of-place version:
function tangent_integrator(ds::DS{false}, Q0;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS)

    D = dimension(ds)
    k = size(Q0)[2]
    initstate = SMatrix{D, k+1}(u0..., Q0...)

    ws_index = SVector{k, Int}((2:k+1)...)

    tangenteom = (u, p, t) -> begin
        du = ds.prob.f(u[:, 1], p, t)
        J = ds.jacobian(u[:, 1], p, t)

        dW = J*u[:, ws_index]
        return hcat(du, dW)
    end

    if problemtype(ds) == ODEProblem
        solver, newkw = extract_solver(diff_eq_kwargs)
        tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, solver; newkw..., save_everystep = false)
    elseif problemtype(ds) == DiscreteProblem
        tanprob = DiscreteProblem(tangenteom, initstate, DDS_TSPAN, ds.prob.p)
        return integ = init(tanprob, FunctionMap(), save_everystep = false)
    end
end
