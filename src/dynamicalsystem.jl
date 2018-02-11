using OrdinaryDiffEq, ForwardDiff, StaticArrays
import OrdinaryDiffEq: isinplace, ODEIntegrator
import Base: eltype

export dimension, state, DynamicalSystem, integrator, tangent_integrator

#######################################################################################
#                          Basic functions and interface                              #
#######################################################################################

dimension(prob::DEProblem) = length(prob.u0)
eltype(prob::DEProblem) = eltype(prob.u0)
state(prob::DEProblem) = prob.u0
hascallback(prob::DEProblem) = prob.callback != nothing
statetype(prob::DEProblem) = eltype(prob.u0)
set_parameter!(prob, index, value) = (prob.p[index] = value)

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

orthonormal(D, k) = qr(rand(D, D))[1][:, 1:k]

#######################################################################################
#                                  DynamicalSystem                                    #
#######################################################################################

struct DynamicalSystem{
        DEP<:DEProblem, # problem type, used in dispatch as well
        IIP, # is in place , for dispatch purposes and clarity
        JAC, # jacobian function (either user-provided or FD)
        IAD} # is auto differentiated? Only for constructing TangentEvolver
    prob::DEP
    jacobian::JAC
end

DS = DynamicalSystem
isautodiff(ds::DS{DEP, IIP, JAC, IAD}) where {DEP, IIP, JAC, IAD} = IAD
problemtype(ds::DS{DEP, IIP, JAC, IAD}) where {DEP<:DiscreteProblem, IIP, JAC, IAD} =
DiscreteProblem
problemtype(ds::DS{DEP, IIP, JAC, IAD}) where {DEP<:ODEProblem, IIP, JAC, IAD} =
ODEProblem

function create_jacobian(prob)
    IIP = isinplace(prob)
    if IIP
        dum = deepcopy(prob.u0)
        cfg = ForwardDiff.JacobianConfig(
            (y, x) -> prob.f(y, x, prob.p, prob.tspan[1]),
            dum, prob.u0)
        jacobian = (J, u, p, t) ->
        ForwardDiff.jacobian!(J, (y, x) -> prob.f(y, x, p, t), dum, u)
        # you cant use the confing if you change the anonymous function...
    else
        cfg = ForwardDiff.JacobianConfig(
            (x) -> prob.f(x, prob.p, prob.tspan[1]), prob.u0)
        jacobian = (u, p, t) ->
        ForwardDiff.jacobian((x) -> prob.f(x, p, t), u)
        # you cant use the confing if you change the anonymous function...
    end
    return jacobian
end

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
    return DynamicalSystem{DEP, IIP, JAC, true}(prob, jacobian)
end
function DynamicalSystem(prob::DEProblem, jacobian::JAC) where {JAC}
    IIP = isinplace(prob)
    JIP = isinplace(jacobian, 4)
    JIP == IIP || throw(ArgumentError(
    "The jacobian function and the equations of motion are not of the same form!"*
    " The jacobian `isinlace` was $(JIP) while the eom `isinplace` was $(IIP)."))
    DEP = typeof(prob)
    return DynamicalSystem{DEP, IIP, JAC, false}(prob, jacobian)
end

CDS = ContinuousDynamicalSystem
DDS = DiscreteDynamicalSystem

# Expand methods
for f in (:isinplace, :dimension, :eltype, :statetype, :state, :systemtype,
    :set_parameter!)
    @eval begin
        @inline ($f)(ds::DynamicalSystem, args...) = $(f)(ds.prob, args...)
    end
end

# Jacobian:
function jacobian(ds::DS{DEP, true, A, B}, u = ds.prob.u0) where {DEP, A, B}
    D = dimension(ds)
    J = similar(u, D, D)
    ds.jacobian(J, u, ds.prob.p, ds.prob.tspan[1])
    return J
end

jacobian(ds::DS{DEP, false, A, B}, u = ds.prob.u0) where {DEP, A, B} =
ds.jacobian(u, ds.prob.p, ds.prob.tspan[1])


#######################################################################################
#                                    Integrators                                      #
#######################################################################################
function integrator(ds::DS{ODE};
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, u0 = ds.prob.u0
    ) where {ODE<:ODEProblem}

    solver, newkw = extract_solver(diff_eq_kwargs)

    prob = ODEProblem(ds.prob.f, u0, CDS_TSPAN, ds.prob.p)
    integ = init(prob, solver; newkw..., save_everystep = false)
end

function integrator(ds::DS{DD};
    u0 = ds.prob.u0) where {DD<:DiscreteProblem}

    prob = DiscreteProblem(ds.prob.f, u0, DDS_TSPAN, ds.prob.p)
    integ = init(prob, FunctionMap(); save_everystep = false)
end


### Tangent integrators ###

# in-place autodifferentiated jacobian helper struct
# (it exists solely to see if we can use JacobianConfig)
struct TangentIIP{F, JM}
    f::F     # original eom
    J::JM    # Jacobian matrix (written in-place)
end
function (j::TangentIIP)(du, u, p, t)
    reducedf = (du, u) -> j.f(du, u, p, t)
    # The following line applies f to du[:,1] and calculates jacobian
    ForwardDiff.jacobian!(j.J, reducedf, view(du, :, 1), view(u, :, 1))
    # This performs dY/dt = J(u) ⋅ Y
    A_mul_B!((@view du[:, 2:end]), j.J, (@view u[:, 2:end]))
    return
    # Notice that this is a temporary solution. It is not performant because
    # it does not use JacobianConfig. But at the moment I couldn't find a way
    # to make it work.
end

# In-place, autodifferentiated version:
function tangent_integrator(ds::DS{ODE, true, JAC, true}, Q0;
    diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS, u0 = ds.prob.u0
    ) where {ODE<:ODEProblem, JAC}

    solver, newkw = extract_solver(diff_eq_kwargs)

    D = dimension(ds)
    tangenteom = TangentIIP(ds.prob.f, similar(ds.prob.u0, D, D))
    initstate = hcat(u0, Q0)
    tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
    integ = init(tanprob, solver; newkw..., save_everystep = false)
end
function tangent_integrator(ds::DS{DD, true, JAC, true}, Q0;
    u0 = ds.prob.u0) where {DD<:DiscreteProblem, JAC}

    D = dimension(ds)
    tangenteom = TangentIIP(ds.prob.f, similar(ds.prob.u0, D, D))
    initstate = hcat(u0, Q0)
    tanprob = DiscreteProblem(tangenteom, initstate, DDS_TSPAN, ds.prob.p)
    integ = init(tanprob, FunctionMap(), save_everystep = false)
end

# In-place version:
function tangent_integrator(ds::DS{ODE, true, JAC, false}, Q0;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS) where
    {ODE<:ODEProblem, JAC}

    D = dimension(ds)
    J = jacobian(ds)
    tangenteom = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        ds.prob.f(view(du, :, 1), uv, p, t)
        ds.jacobian(J, uv, p, t)
        A_mul_B!((@view du[:, 2:end]), J, (@view u[:, 2:end]))
        nothing
    end

    solver, newkw = extract_solver(diff_eq_kwargs)
    initstate = hcat(u0, Q0)
    tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
    integ = init(tanprob, solver; newkw..., save_everystep = false)
end
function tangent_integrator(ds::DS{DD, true, JAC, false}, Q0;
    u0 = ds.prob.u0) where {DD<:DiscreteProblem, JAC}

    D = dimension(ds)
    J = jacobian(ds)
    tangenteom = (du, u, p, t) -> begin
        uv = @view u[:, 1]
        ds.prob.f(view(du, :, 1), uv, p, t)
        ds.jacobian(J, uv, p, t)
        A_mul_B!((@view du[:, 2:end]), J, (@view u[:, 2:end]))
        nothing
    end
    initstate = cat(2, u0, Q0)
    tanprob = DiscreteProblem(tangenteom, initstate, DDS_TSPAN, ds.prob.p)
    integ = init(tanprob, FunctionMap(), save_everystep = false)
end

# out-of-place version, same regardless of autodiff:
function tangent_integrator(ds::DS{ODE, false, JAC, IAD}, Q0;
    u0 = ds.prob.u0, diff_eq_kwargs = DEFAULT_DIFFEQ_KWARGS) where
    {ODE<:ODEProblem, JAC, IAD}

    D = dimension(ds)
    k = size(Q0)[2]
    ws_index = SVector{k, Int}((2:k+1)...)
    tangenteom = (u, p, t) -> begin
        du = ds.prob.f(u[:, 1], p, t)
        J = ds.jacobian(u, p, t)

        dW = J*u[:, ws_index]
        return hcat(du, dW)
    end

    solver, newkw = extract_solver(diff_eq_kwargs)
    initstate = SMatrix{D, k+1}(u0..., Q0...)
    tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
    integ = init(tanprob, solver; newkw..., save_everystep = false)
end
function tangent_integrator(ds::DS{DP, false, JAC, IAD}, Q0;
    u0 = ds.prob.u0) where
    {DO<:DiscreteProblem, JAC, IAD}

    D = dimension(ds)
    k = size(Q0)[2]
    ws_index = SVector{k, Int}((2:k+1)...)
    tangenteom = (u, p, t) -> begin
        du = ds.prob.f(u[:, 1], p, t)
        J = ds.jacobian(u, p, t)

        dW = J*u[:, ws_index]
        return hcat(du, dW)
    end

    solver, newkw = extract_solver(diff_eq_kwargs)
    initstate = SMatrix{D, k+1}(u0..., Q0...)
    tanprob = ODEProblem(tangenteom, initstate, CDS_TSPAN, ds.prob.p)
    integ = init(tanprob, FunctionMap(); save_everystep = false)
end



#####################################################################################
#                                   Set-State                                       #
#####################################################################################

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
