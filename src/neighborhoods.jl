using NearestNeighbors, StaticArrays
using Distances: Euclidean
import NearestNeighbors: KDTree

export AbstractNeighborhood
export FixedMassNeighborhood, FixedSizeNeighborhood
export neighborhood, KDTree

#####################################################################################
#                              Neighborhoods n stuff                                #
#####################################################################################
"""
    AbstractNeighborhood
Supertype of methods for deciding the neighborhood of points for a given point.

Concrete subtypes:
* `FixedMassNeighborhood(K::Int)`  : The neighborhood of a point consists of the `K`
  nearest neighbors of the point.
* `FixedSizeNeighborhood(ε::Real)` : The neighborhood of a point consists of all
  neighbors that have distance < `ε` from the point.

Notice that these distances are always computed using the Euclidean distance
in `D`-dimensional space.

See also [`neighborhood`](@ref).
"""
abstract type AbstractNeighborhood end
struct FixedMassNeighborhood <: AbstractNeighborhood
    K::Int
end
FixedMassNeighborhood() = FixedMassNeighborhood(1)
struct FixedSizeNeighborhood <: AbstractNeighborhood
    ε::Float64
end
FixedSizeNeighborhood() = FixedSizeNeighborhood(0.001)

"""
    neighborhood([n,] point, tree::KDTree, method::AbstractNeighborhood)
Return a vector of indices which are the neighborhood of `point`.
`n` is the index of the `point` in the original dataset. Do not pass any index
if the `point` is not part of the dataset.

If the original dataset is `data <: AbstractDataset`, then
use `tree = KDTree(data)` to obtain the `tree` instance (which also
contains a copy of the data).

The `method` can be a subtype of [`AbstractNeighborhood`](@ref).

`neighborhood` works for *any* subtype of `AbstractDataset`.

## References

`neighborhood` simply interfaces the functions
`knn` and `inrange` from
[NearestNeighbors.jl](https://github.com/KristofferC/NearestNeighbors.jl) by using
the last argument, `method`.
"""
function neighborhood(
    n::Int, point::AbstractVector, tree::KDTree, method::FixedMassNeighborhood)
    idxs, = knn(tree, point, method.K, false, i -> i==n)
    return idxs
end
function neighborhood(
    n::Int, point::AbstractVector, tree::KDTree, method::FixedSizeNeighborhood)
    idxs = inrange(tree, point, method.ε)
    deleteat!(idxs, findin(idxs, n)) # unfortunately this has to be done...
    return idxs
end
function neighborhood(
    point::AbstractVector, tree::KDTree, method::FixedMassNeighborhood)
    idxs, = knn(tree, point, method.K, false)
    return idxs
end
function neighborhood(
    point::AbstractVector, tree::KDTree, method::FixedSizeNeighborhood)
    idxs = inrange(tree, point, method.ε)
    return idxs
end

KDTree(D::AbstractDataset) = KDTree(D.data, Euclidean())
