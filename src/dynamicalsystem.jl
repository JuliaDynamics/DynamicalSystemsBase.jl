using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, ODEIntegrator, step!
import Base: eltype

export dimension, state, DynamicalSystem, DS, integrator, tangent_integrator
export ContinuousDynamicalSystem, CDS, DiscreteDynamicalSystem, DDS
export set_parameter!, step!

#######################################################################################
#                          Basic functions and interface                              #
#######################################################################################

dimension(prob::DEProblem) = length(prob.u0)
eltype(prob::DEProblem) = eltype(prob.u0)
state(prob::DEProblem) = prob.u0
hascallback(prob::DEProblem) = prob.callback != nothing
statetype(prob::DEProblem) = eltype(prob.u0)

systemtype(::ODEProblem) = "continuous"
systemtype(::DiscreteProblem) = "discrete"

function extract_solver(diff_eq_kwargs)
    # Extract solver from kwargs
    if haskey(diff_eq_kwargs, :solver)
        newkw = deepcopy(diff_eq_kwargs)
        solver = diff_eq_kwargs[:solver]
        pop!(newkw, :solver)
    else
        solver = DEFAULT_SOLVER
        newkw = diff_eq_kwargs
    end
    return solver, newkw
end

const DEFAULT_DIFFEQ_KWARGS = Dict{Symbol, Any}(:abstol => 1e-9, :reltol => 1e-9)
const DEFAULT_SOLVER = Vern9()
const DDS_TSPAN = (0, Int(1e6))
const CDS_TSPAN = (0.0, Inf)

function step!(integ, Δt::Real)
    t = integ.t
    while integ.t < t + Δt
        step!(integ)
    end
end

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################
"""
    DynamicalSystem ≡ DS

## Constructors

```julia
CDS(eom, state::AbstractVector, p [, jacobian])
DDS(eom, state::AbstractVector, p [, jacobian])
DS(prob [, jacobian])
```
We don't use/care about `tspan`.
"""
struct DynamicalSystem{
        IIP, # is in place , for dispatch purposes and clarity
        IAD, # is auto differentiated? Only for constructing tangent_integrator
        PT<:DEProblem, # problem type
        JAC} # jacobian function (either user-provided or FD)
    prob::PT
    jacobian::JAC
end

DS = DynamicalSystem
isautodiff(ds::DS{IIP, IAD, DEP, JAC}) where {DEP, IIP, JAC, IAD} = IAD
problemtype(ds::DS{IIP, IAD, DEP, JAC}) where {DEP<:DiscreteProblem, IIP, JAC, IAD} =
DiscreteProblem
problemtype(ds::DS{IIP, IAD, DEP, JAC}) where {DEP<:ODEProblem, IIP, JAC, IAD} =
ODEProblem

"""
    ContinuousDynamicalSystem ≡ CDS
Alias of `DynamicalSystem` restricted to continuous systems (also called *flows*).

See [`DynamicalSystem`](@ref) for constructors.
"""
ContinuousDynamicalSystem{IIP, IAD, PT, JAC} =
DynamicalSystem{IIP, IAD, PT, JAC} where {IIP, IAD, PT<:ODEProblem, JAC}

"""
    DiscreteDynamicalSystem ≡ DDS
Alias of `DynamicalSystem` restricted to discrete systems (also called *maps*).

See [`DynamicalSystem`](@ref) for constructors.
"""
DiscreteDynamicalSystem{IIP, IAD, PT, JAC} =
DynamicalSystem{IIP, IAD, PT, JAC} where {IIP, IAD, PT<:DiscreteProblem, JAC}

CDS = ContinuousDynamicalSystem
DDS = DiscreteDynamicalSystem

# High level constructors (you can't deduce if system is discrete
# or continuous from just the equations of motion!)
function ContinuousDynamicalSystem(eom, state::AbstractVector, p, j = nothing)
    IIP = isinplace(eom, 4)
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    prob = ODEProblem(eom, u0, CDS_TSPAN, p)
    if j == nothing
        return DS(prob)
    else
        return DS(prob, j)
    end
end
function DiscreteDynamicalSystem(eom, state::AbstractVector, p, j = nothing)
    IIP = isinplace(eom, 4)
    u0 = IIP ? Vector(state) : SVector{length(state)}(state...)
    prob = DiscreteProblem(eom, u0, DDS_TSPAN, p)
    if j == nothing
        return DS(prob)
    else
        return DS(prob, j)
    end
end
function DynamicalSystem(prob::DEProblem)
    IIP = isinplace(prob)
    jacobian = create_jacobian(prob)
    DEP = typeof(prob)
    JAC = typeof(jacobian)
    return DynamicalSystem{IIP, true, DEP, JAC}(prob, jacobian)
end
function DynamicalSystem(prob::DEProblem, jacobian::JAC) where {JAC}
    IIP = isinplace(prob)
    JIP = isinplace(jacobian, 4)
    JIP == IIP || throw(ArgumentError(
    "The jacobian function and the equations of motion are not of the same form!"*
    " The jacobian `isinlace` was $(JIP) while the eom `isinplace` was $(IIP)."))
    DEP = typeof(prob)
    return DynamicalSystem{IIP, false, DEP, JAC}(prob, jacobian)
end



# Expand methods
for f in (:isinplace, :dimension, :eltype, :statetype, :state, :systemtype,
    :set_parameter!)
    @eval begin
        @inline ($f)(ds::DynamicalSystem, args...) = $(f)(ds.prob, args...)
    end
end


#####################################################################################
#                                    Jacobians                                      #
#####################################################################################
function create_jacobian(prob)
    IIP = isinplace(prob)
    if IIP
        dum = deepcopy(prob.u0)
        cfg = ForwardDiff.JacobianConfig(
            (y, x) -> prob.f(y, x, prob.p, prob.tspan[1]),
            dum, prob.u0)
        jacobian = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> prob.f(y, x, p, t),
        dum, u, cfg, Val{false}())
    else
        # SVector methods do *not* use the config
        # cfg = ForwardDiff.JacobianConfig(
        #     (x) -> prob.f(x, prob.p, prob.tspan[1]), prob.u0)
        jacobian = (u, p, t) ->
        ForwardDiff.jacobian((x) -> prob.f(x, p, t), u, #=cfg=#)
    end
    return jacobian
end
# Jacobian:
function jacobian(ds::DS{true}, u = ds.prob.u0)
    D = dimension(ds)
    J = similar(u, D, D)
    ds.jacobian(J, u, ds.prob.p, ds.prob.tspan[1])
    return J
end

jacobian(ds::DS{false}, u = ds.prob.u0) =
ds.jacobian(u, ds.prob.p, ds.prob.tspan[1])

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################

Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds)*" dynamical system"

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(ds.jacobian)"

function Base.show(io::IO, ds::DS)
    ps = 12
    text = summary(ds)
    print(io, text*"\n",
    rpad(" state: ", ps)*"$(state(ds))\n",
    rpad(" e.o.m.: ", ps)*"$(ds.prob.f)\n",
    rpad(" in-place? ", ps)*"$(isinplace(ds))\n",
    rpad(" jacobian: ", ps)*"$(jacobianstring(ds))\n"
    )
end


#######################################################################################
#                                    Integrators                                      #
#######################################################################################
safe_state_type(ds::DS{true}, u0) = typeof(u0) <: Vector ? u0 : Vector(u0)
function safe_state_type(ds::DS{false}, u0)
    typeof(u0) <: SVector ? u0 : SVector{dimension(ds)}(u0...)
end


"""
    integrator(DS, u0 = ds.prob.u0)
"""
function integrator(ds::CDS, u0 = ds.prob.u0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS
    )

    U0 = safe_state_type(ds, u0)
    solver, newkw = extract_solver(diff_eq_kwargs)

    prob = ODEProblem(ds.prob.f, U0, CDS_TSPAN, ds.prob.p)
    integ = init(prob, solver; newkw..., save_everystep = false)
end

function integrator(ds::DDS, u0 = ds.prob.u0)
    U0 = safe_state_type(ds, u0)
    prob = DiscreteProblem(ds.prob.f, U0, DDS_TSPAN, ds.prob.p)
    integ = init(prob, FunctionMap(); save_everystep = false)
end


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




#####################################################################################
#                                    Auxilary                                       #
#####################################################################################
"""
    set_parameter!(ds::DynamicalSystem, index, value)
    set_parameter!(ds::DynamicalSystem, values)
Change one or many parameters of the system
by setting `p[index] = value` in the first case
and `p .= values` in the second.
"""
set_parameter!(prob, index, value) = (prob.p[index] = value)
set_parameter!(prob, values) = (prob.p .= values)
set_parameter!(ds::DS, args...) = set_parameter!(ds.prob, args...)
