#####################################################################################
#                                Pairwse Distance                                   #
#####################################################################################
using NearestNeighbors, StaticArrays, LinearAlgebra
export min_pairwise_distance, orthonormal

# min_pairwise_distance contributed by Kristoffer Carlsson
"""
    min_pairwise_distance(data) -> (min_pair, min_dist)
Calculate the minimum pairwise distance in the data (`Matrix`, `Vector{Vector}` or
`Dataset`). Return the index pair
of the datapoints that have the minimum distance, as well as its value.
"""
function min_pairwise_distance(cts::AbstractMatrix)
    if size(cts, 1) > size(cts, 2)
        error("Points must be close (transpose the Matrix)")
    end
    tree = KDTree(cts)
    min_d = Inf
    min_pair = (0, 0)
    for p in 1:size(cts, 2)
        inds, dists = knn(tree, view(cts, :, p), 1, false, i -> i == p)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (p, ind)
        end
    end
    return min_pair, min_d
end

min_pairwise_distance(d::AbstractDataset) = min_pairwise_distance(d.data)

function min_pairwise_distance(
    pts::Vector{SVector{D,T}}) where {D,T<:Real}
    tree = KDTree(pts)
    min_d = eltype(pts[1])(Inf)
    min_pair = (0, 0)
    for p in 1:length(pts)
        inds, dists = knn(tree, pts[p], 1, false, i -> i == p)
        ind, dist = inds[1], dists[1]
        if dist < min_d
            min_d = dist
            min_pair = (p, ind)
        end
    end
    return min_pair, min_d
end

#####################################################################################
#                                Conversions                                        #
#####################################################################################
to_matrix(a::AbstractVector{<:AbstractVector}) = cat(2, a...)
to_matrix(a::AbstractMatrix) = a
function to_Smatrix(m)
    M = to_matrix(m)
    a, b = size(M)
    return SMatrix{a, b}(M)
end
to_vectorSvector(a::AbstractVector{<:SVector}) = a
function to_vectorSvector(a::AbstractMatrix)
    S = eltype(a)
    D, k = size(a)
    ws = Vector{SVector{D, S}}(k)
    for i in 1:k
        ws[i] = SVector{D, S}(a[:, i])
    end
    return ws
end

"""
    orthonormal(D, k) -> ws
Return a matrix `ws` with `k` columns, each being
an `D`-dimensional orthonormal vector.

Always returns `SMatrix` for stability reasons.
"""
function orthonormal end

@inline function orthonormal(D::Int, k::Int)
    k > D && throw(ArgumentError("k must be ≤ D"))
    q = qr(rand(SMatrix{D, k})).Q
end
