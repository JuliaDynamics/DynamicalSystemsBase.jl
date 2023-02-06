export PlaneCrossing, poincaresos

# TODO: Remove keyword `save_idxs`!

####################################################################################################
# Hyperplane definitions
####################################################################################################
"""
    PlaneCrossing(plane, dir) → pc

Create a struct that can be called as a function `pc(u)` that returns the signed distance
of state `u` from the hyperplane `plane` (positive means in front of the hyperplane).
See [`PoincareMap`](@ref) for what `plane` can be (tuple or vector).
"""
struct PlaneCrossing{P, D, T}
    plane::P
    dir::Bool
    n::SVector{D, T}  # normal vector
    p₀::SVector{D, T} # arbitrary point on plane
end
PlaneCrossing(plane::Tuple, dir) = PlaneCrossing(plane, dir, SVector(true), SVector(true))
function PlaneCrossing(plane::AbstractVector, dir)
    n = plane[1:end-1] # normal vector to hyperplane
    i = findfirst(!iszero, plane)
    D = length(plane)-1; T = eltype(plane)
    p₀ = zeros(D)
    p₀[i] = plane[end]/plane[i] # p₀ is an arbitrary point on the plane.
    PlaneCrossing(plane, dir, SVector{D, T}(n), SVector{D, T}(p₀))
end

# Definition of functional behavior: signed distance from hyperplane
function (hp::PlaneCrossing{P})(u::AbstractVector) where {P<:Tuple}
    @inbounds x = u[hp.plane[1]] - hp.plane[2]
    hp.dir ? x : -x
end
function (hp::PlaneCrossing{P})(u::AbstractVector) where {P<:AbstractVector}
    x = zero(eltype(u))
    D = length(u)
    @inbounds for i in 1:D
        x += u[i]*hp.plane[i]
    end
    @inbounds x -= hp.plane[D+1]
    hp.dir ? x : -x
end


"Ensure hyperplane is matching with dimension `D`."
function check_hyperplane_match(plane, D)
    P = typeof(plane)
    L = length(plane)
    if P <: AbstractVector
        if L != D + 1
            throw(ArgumentError(
            "The plane for the `poincaresos` must be either a 2-Tuple or a vector of "*
            "length D+1 with D the dimension of the system."
            ))
        end
    elseif P <: Tuple
        if !(P <: Tuple{Int, Number})
            throw(ArgumentError(
            "If the plane for the `poincaresos` is a 2-Tuple then "*
            "it must be subtype of `Tuple{Int, Number}`."
            ))
        end
    else
        throw(ArgumentError(
        "Unrecognized type for the `plane` argument."
        ))
    end
end

##############################################################################################
# Poincare Section for Datasets (trajectories)
##############################################################################################
# TODO: Nice improvement would be to use cubic interpolation instead of linear,
# using points i-2, i-1, i, i+1
"""
    poincaresos(A::AbstractDataset, plane; kwargs...) → P::Dataset

Calculate the Poincaré surface of section of the given dataset with the given `plane`
by performing linear interpolation betweeen points that sandwich the hyperplane.

Argument `plane` and keywords `direction, warning, save_idxs`
are the same as in the method below.
"""
function poincaresos(A::AbstractDataset, plane;
        direction = -1, warning = true, save_idxs = dimension(A)
    )
    check_hyperplane_match(plane, size(A, 2))
    i = typeof(save_idxs) <: Int ? save_idxs : SVector{length(save_idxs), Int}(save_idxs...)
    planecrossing = PlaneCrossing(plane, direction > 0)
    data = poincaresos(A, planecrossing, i)
    warning && length(data) == 0 && @warn PSOS_ERROR
    return Dataset(data)
end
function poincaresos(A::Dataset, planecrossing::PlaneCrossing, j)
    i, L = 1, length(A)
    data = _initialize_output(A[1], j)
    # Check if initial condition is already on the plane
    planecrossing(A[i]) == 0 && push!(data, A[i][j])
    i += 1
    side = planecrossing(A[i])

    while i ≤ L # We always check point i vs point i-1
        while side < 0 # bring trajectory infront of hyperplane
            i == L && break
            i += 1
            side = planecrossing(A[i])
        end
        while side ≥ 0 # iterate until behind the hyperplane
            i == L && break
            i += 1
            side = planecrossing(A[i])
        end
        i == L && break
        # It is now guaranteed that A crosses hyperplane between i-1 and i
        ucross = interpolate_crossing(A[i-1], A[i], planecrossing)
        push!(data, ucross[j])
    end
    return data
end

function interpolate_crossing(A, B, pc::PlaneCrossing{<:AbstractVector})
    # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    t = LinearAlgebra.dot(pc.n, (pc.p₀ .- A))/LinearAlgebra.dot((B .- A), pc.n)
    return A .+ (B .- A) .* t
end

function interpolate_crossing(A, B, pc::PlaneCrossing{<:Tuple})
    # https://en.wikipedia.org/wiki/Linear_interpolation
    y₀ = A[pc.plane[1]]; y₁ = B[pc.plane[1]]; y = pc.plane[2]
    t = (y - y₀) / (y₁ - y₀) # linear interpolation with t₀ = 0, t₁ = 1
    return A .+ (B .- A) .* t
end
