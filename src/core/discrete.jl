using StaticArrays, ForwardDiff

#####################################################################################
#                            DiscreteIntegrator                                    #
#####################################################################################
mutable struct MinimalDiscreteIntegrator{IIP, S, D, F, P}
    f::F      # integrator f
    u::S      # integrator state
    t::Int    # integrator "time" (counter)
    dummy::S  # dummy, used only in the IIP version
    p::P      # parameter container, I don't know why
    t0::Int   # initial time (only for reinit!)
end
const MDI = MinimalDiscreteIntegrator
isinplace(::MDI{IIP}) where {IIP} = IIP
stateeltype(::MDI{IIP, S}) where {IIP, S} = eltype(S)
stateeltype(::MDI{IIP, S}) where {IIP, S<:Vector{<:AbstractArray{T}}} where {T} = T
isdiscretetime(p::MinimalDiscreteIntegrator) = true
DelayEmbeddings.dimension(::MDI{IIP, S, D}) where {IIP, S, D} = D

function reinit!(integ::MDI, u = integ.u, Q0 = nothing; t0 = integ.t0)
    integ.u = u
    integ.dummy = copy(u)
    integ.t = t0
    if Q0 !== nothing
        set_deviations!(integ, Q0)
    end
    return
end

function (integ::MinimalDiscreteIntegrator)(t::Real)
    if t == integ.t
        return integ.u
    else
        error("Can't extrapolate discrete systems!")
    end
end

# Get state for parallel:
get_state(a::MDI{IIP, S}) where {IIP, S<:Vector{<:AbstractVector}} = a.u[1]
get_state(a::MDI{IIP, S}, k) where {IIP, S<:Vector{<:AbstractVector}} = a.u[k]
function set_state!(
    integ::MDI{Alg, S}, u::AbstractVector, k::Int = 1
    ) where {Alg, S<:Vector{<:AbstractVector}}
    integ.u[k] = u
end

# for autodiffed in-place version (where state is Matrix)
get_state(integ::MDI{Alg, S}) where {Alg, S<:AbstractMatrix} = integ.u[:, 1]
set_state!(integ::MDI{Alg, S}, u) where {Alg, S<:AbstractMatrix} = (integ.u[:, 1] .= u)
get_deviations(integ::MDI{Alg, S}) where {Alg, S<:AbstractMatrix} =
    @view integ.u[:, 2:end]
set_deviations!(integ::MDI{Alg, S}, Q) where {Alg, S<:AbstractMatrix} =
    (integ.u[:, 2:end] = Q)

#####################################################################################
#                                   Stepping                                        #
#####################################################################################
# IIP version
function step!(integ::MDI{true})
    # vector swap
    integ.dummy, integ.u = integ.u, integ.dummy
    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.t += 1
    return
end
function step!(integ::MDI{true}, N)
    for i in 1:N
        integ.dummy, integ.u = integ.u, integ.dummy
        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.t += 1
    end
    return
end

# OOP version
step!(integ::MDI{false}) =
(integ.u = integ.f(integ.u, integ.p, integ.t); integ.t +=1; nothing)
function step!(integ::MDI{false}, N)
    for i in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        integ.t += 1
    end
    return
end

u_modified!(integ::MDI, ::Bool) = nothing

step!(integ::MDI, N, stop_at_tdt) = step!(integ, N)

#####################################################################################
#                           TangentDiscreteIntegrator                               #
#####################################################################################
# For discrete systems a special, super-performant version of tangent integrator
# can exist. We make this here.
mutable struct TangentDiscreteIntegrator{IIP, S, D, F, P, JAC, JM, WM}
    f::F            # integrator f
    u::S            # integrator state
    t::Int          # integrator "time" (counter)
    dummy::S        # dummy, used only in the IIP version
    p::P            # parameter container, I don't know why
    t0::Int         # initial time (only for reinit!)
    jacobian::JAC   # jacobian function
    J::JM           # jacobian matrix
    W::WM           # tangent vectors (in form of matrix)
    dummyW::WM      # dummy, only used in IIP version
end

const TDI = TangentDiscreteIntegrator
stateeltype(::TDI{IIP, S}) where {IIP, S} = eltype(S)
isdiscretetime(p::TangentDiscreteIntegrator) = true
DelayEmbeddings.dimension(::TDI{IIP, S, D}) where {IIP, S, D} = D

u_modified!(t::TDI, a) = nothing

# set_state is same as in standard, see dynamicalsystem.jl

get_deviations(t::TDI) = t.W
set_deviations!(t::TDI, Q) = (t.W = Q) # works for both Matrix/SMatrix Q
function set_deviations!(t::TDI, Q::LinearAlgebra.AbstractQ)
    # This method exists because there is a more efficient way to copy columns of Q
    # without transforming `Q` into a matrix
    # integ.u[:, 2:end] .= Matrix(Q) <- this is bad, it allocates!
    copyto!(t.W, I)
    lmul!(Q, t.W)
end

function reinit!(integ::TDI, u = integ.u, Q0 = nothing; t0 = integ.t0)
    set_state!(integ, u)
    if Q0 !== nothing
        set_deviations!(integ, Q0)
    end
    integ.t = t0
    return
end


#####################################################################################
#                                 Tangent Stepping                                  #
#####################################################################################
function step!(integ::TDI{true})
    integ.dummy, integ.u = integ.u, integ.dummy
    integ.dummyW, integ.W = integ.W, integ.dummyW

    integ.f(integ.u, integ.dummy, integ.p, integ.t)
    integ.jacobian(integ.J, integ.u, integ.p, integ.t)

    mul!(integ.W, integ.J, integ.dummyW)
    integ.t += 1
    return
end
function step!(integ::TDI{true}, N::Real)
    for i in 1:N
        integ.dummy, integ.u = integ.u, integ.dummy
        integ.dummyW, integ.W = integ.W, integ.dummyW

        integ.f(integ.u, integ.dummy, integ.p, integ.t)
        integ.jacobian(integ.J, integ.u, integ.p, integ.t)

        mul!(integ.W, integ.J, integ.dummyW)
        integ.t += 1
    end
    return
end

function step!(integ::TDI{false})
    integ.u = integ.f(integ.u, integ.p, integ.t)
    J = integ.jacobian(integ.u, integ.p, integ.t)
    integ.W = J*integ.W
    integ.t += 1
    return
end
function step!(integ::TDI{false}, N::Real)
    for i in 1:N
        integ.u = integ.f(integ.u, integ.p, integ.t)
        J = integ.jacobian(integ.u, integ.p, integ.t)
        integ.W = J*integ.W
        integ.t += 1
    end
    return
end

step!(integ::TDI, N, stop_at_tdt) = step!(integ, N)

#####################################################################################
#                            Creating Integrators                                   #
#####################################################################################
function integrator(ds::DDS{IIP, S, D, F, P}, u0 = ds.u0; kwargs...) where {IIP, S, D, F, P}
    return MinimalDiscreteIntegrator{IIP, S, D, F, P}(
    ds.f, S(u0), ds.t0, S(copy(u0)), ds.p, ds.t0)
end

function tangent_integrator(ds::DDS, k::Int = dimension(ds); kwargs...)
    return tangent_integrator(
    ds, orthonormal(dimension(ds), k); kwargs...)
end

function tangent_integrator(ds::DDS{IIP, S, D, F, P, JAC, JM}, Q0::AbstractMatrix;
    u0 = ds.u0, kwargs...) where {IIP, S, D, F, P, JAC, JM}

    Q = safe_matrix_type(Val{IIP}(), Q0)
    s = safe_state_type(Val{IIP}(), u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    WM = typeof(Q)

    return TDI{IIP, S, D, F, P, JAC, JM, WM}(ds.f, s, ds.t0, deepcopy(s),
    ds.p, ds.t0, ds.jacobian::JAC, deepcopy(ds.J), Q, deepcopy(Q))
end

# Auto-diffed in-place version
function tangent_integrator(ds::DDS{true, S, D, F, P, JAC, JM, true},
    Q0::AbstractMatrix;
    u0 = ds.u0, kwargs...) where {S, D, F, P, JAC, JM}

    t0 = ds.t0
    R = D + length(Q0)
    k = size(Q0)[2]
    Q = safe_matrix_type(Val{true}(), Q0)
    u = safe_state_type(Val{true}(), u0)
    size(Q)[2] > dimension(ds) && throw(ArgumentError(
    "It is not possible to evolve more tangent vectors than the system's dimension!"
    ))

    tangentf = create_tangent_iad(ds.f, ds.J, u, ds.p, t0, Val{k}())

    TF = typeof(tangentf)

    s = hcat(u, Q)
    SS = typeof(s)

    return MDI{true, SS, R, TF, P}(tangentf, s, t0, deepcopy(s), ds.p, t0)
end

function parallel_integrator(ds::DDS{IIP, S, D}, states; kwargs...) where {IIP, S, D}
    peom, st = create_parallel(ds, states)
    F = typeof(peom); X = typeof(st); P = typeof(ds.p)
    return MDI{true, X, D, F, P}(peom, st, ds.t0, deepcopy(st), ds.p, ds.t0)
end

#####################################################################################
#                                 Trajectory                                        #
#####################################################################################
function trajectory(
        ds::DDS{IIP, S, D}, t, u = ds.u0;
        Δt = 1, Ttr = 0, save_idxs = nothing, kwargs...
    ) where {IIP, S, D}
    a = svector_access(save_idxs)
    integ = integrator(ds, u)
    trajectory(integ, t; Δt, Ttr, a)
end

# generic dispatch
function trajectory(integ, args...; kwargs...)
    if isdiscretetime(integ)
        return trajectory_discrete(integ, args...; kwargs...)
    else
        return trajectory_continuous(integ, args...; kwargs...)
    end
end

# This version of trajectory is for any discrete integrator: Poincare map,
# stroboscopic map, or standard MinimalDiscreteIntegrator.
function trajectory_discrete(integ, t, u0 = nothing; Δt=1, Ttr=0, a=nothing)
    !isnothing(u0) && reinit!(integ, u0)
    Δt = round(Int, Δt)
    T = eltype(get_state(integ))
    t0 = current_time(integ)
    tvec = (t0+Ttr):Δt:(t0+t+Ttr)
    L = length(tvec)
    T = eltype(get_state(integ))
    X = isnothing(a) ? dimension(integ) : length(a)
    data = Vector{SVector{X, T}}(undef, L)
    Ttr ≠ 0 && step!(integ, Ttr)
    data[1] = obtain_access(get_state(integ), a)
    for i in 2:L
        step!(integ, Δt)
        data[i] = SVector{X, T}(obtain_access(get_state(integ), a))
    end
    return Dataset(data)
end

# One-dimensional version → Vector
function trajectory(
        ds::DDS{false, S, 1}, t, u = ds.u0;
    	Δt = 1, Ttr = 0, kwargs...
    ) where {S}
    Δt = round(Int, Δt)
    ti = ds.t0
    tvec = (ti+Ttr):Δt:(t+ti+Ttr)
    L = length(tvec)
    integ = integrator(ds, u)
    data = Vector{S}(undef, L)
    Ttr != 0 && step!(integ, Ttr)
    data[1] = integ.u
    for i in 2:L
        step!(integ, Δt)
        data[i] = integ.u
    end
    return data
end

#####################################################################################
#                                Pretty-Printing                                    #
#####################################################################################
Base.summary(ds::MDI) =
"Discrete integrator (e.o.m.: $(eomstring(ds.f)))"
Base.summary(ds::MDI{true, S}) where {S<:Vector{<:AbstractArray}} =
"Discrete parallel integrator with $(length(ds.u)) states "*
"(e.o.m.: $(eomstring(ds.f)))"

function Base.show(io::IO, ds::MDI)
    ps = 3
    text = summary(ds)
    println(io, text)
    println(io, rpad(" t: ", ps)*"$(ds.t)")
    prefix = rpad(" u: ", ps)
    print(io, prefix); printlimited(io, get_state(ds)', Δx = length(prefix)); print(io, "\n")
end
function Base.show(io::IO, ds::MDI{true, S}) where {S<:Vector{<:AbstractArray}}
    ps = 3
    text = summary(ds)
    print(io, text*"\n",
    rpad(" t: ", ps)*"$(ds.t)\n",
    rpad(" states: ", ps)*"\n"
    )
    s = sprint(io -> show(IOContext(io, :limit=>true),
    MIME"text/plain"(), ds.u))
    s = join(split(s, '\n')[2:end], '\n')
    print(io, s)
end

Base.summary(ds::TDI) =
"Discrete tangent-space integrator "*
"(e.o.m.: $(eomstring(ds.f)), jacobian: $(eomstring(ds.jacobian)))"

function Base.show(io::IO, ds::TDI)
    ps = 3
    text = summary(ds)
    println(io, text)
    println(io, rpad(" t: ", ps)*"$(ds.t)")
    prefix = rpad(" u: ", ps)
    print(io, prefix); printlimited(io, get_state(ds)', Δx = length(prefix)); print(io, "\n")
    println(io, " deviation vectors:")
    s = sprint(io -> show(IOContext(io, :limit=>true),
    MIME"text/plain"(), get_deviations(ds)))
    s = join(split(s, '\n')[2:end], '\n')
    print(io, s)
end
