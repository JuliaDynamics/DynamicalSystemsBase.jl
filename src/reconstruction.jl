using StaticArrays
using Base: @_inline_meta
export reconstruct, DelayEmbedding, AbstractEmbedding, MTDelayEmbedding

#####################################################################################
#                        Delay Embedding Reconstruction                             #
#####################################################################################
"""
    AbstractEmbedding
Super-type of embedding methods. Use `subtypes(AbstractEmbedding)` for available
methods.
"""
abstract type AbstractEmbedding <: Function end

"""
    DelayEmbedding(D, τ) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given a timeseries and some index. Calling
```julia
embedding(s, n)
```
will create the `n`-th reconstructed vector of the embedded space, which has `D`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

*Be very careful when choosing `n`, because `@inbounds` is used internally.*
"""
struct DelayEmbedding{D} <: AbstractEmbedding
    delays::SVector{D, Int}
end

@inline DelayEmbedding(D, τ) = DelayEmbedding(Val{D}(), τ)
@inline function DelayEmbedding(::Val{D}, τ::Int) where {D}
    idxs = [k*τ for k in 1:D]
    return DelayEmbedding{D}(SVector{D, Int}(idxs...))
end
@inline function DelayEmbedding(::Val{D}, τ::AbstractVector) where {D}
    D != length(τ) && throw(ArgumentError(
    "Delay time vector length must equal the number of temporal neighbors."
    ))
    return DelayEmbedding{D}(SVector{D, Int}(τ...))
end

@generated function (r::DelayEmbedding{D})(s::AbstractArray{T}, i) where {D, T}
    gens = [:(s[i + r.delays[$k]]) for k=1:D]
    quote
        @_inline_meta
        @inbounds return SVector{$D+1,T}(s[i], $(gens...))
    end
end

"""
    reconstruct(s, D, τ)
Reconstruct `s` using the delay coordinates embedding with `D` temporal neighbors
and delay `τ` and return the result as a [`Dataset`](@ref).

## Description
### Single Timeseries
If `τ` is an integer, then the ``n``-th entry of the embedded space is
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+D\\tau))
```
If instead `τ` is a vector of integers, so that `length(τ) == D`,
then the ``n``-th entry is
```math
(s(n), s(n+\\tau[1]), s(n+\\tau[2]), \\dots, s(n+\\tau[D]))
```

The reconstructed dataset can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].
The case of different delay times allows reconstructing systems with many time scales,
see [3].

*Notice* - The dimension of the returned dataset is `D+1`!

### Multiple Timeseries
To make a reconstruction out of a multiple timeseries (i.e. trajectory) the number
of timeseries must be known by type, so `s` can be either:

* `s::AbstractDataset{B}`
* `s::SizedAray{A, B}`

If the trajectory is for example ``(x, y)`` and `τ` is integer, then the ``n``-th
entry of the embedded space is
```math
(x(n), y(n), x(n+\\tau), y(n+\\tau), \\dots, x(n+D\\tau), y(n+D\\tau))
```
If `τ` is an `AbstractMatrix{Int}`, so that `size(τ) == (D, B)`,
then we have
```math
(x(n), y(n), x(n+\\tau[1, 1]), y(n+\\tau[1, 2]), \\dots, x(n+\\tau[D, 1]), y(n+\\tau[D, 2]))
```

*Notice* - The dimension of the returned dataset is `(D+1)*B`!

## References
[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)

[3] : K. Judd & A. Mees, [Physica D **120**, pp 273 (1998)](https://www.sciencedirect.com/science/article/pii/S0167278997001188)
"""
@inline function reconstruct(s::AbstractVector{T}, D, τ) where {T}
    de::DelayEmbedding{D} = DelayEmbedding(Val{D}(), τ)
    L = length(s) - maximum(de.delays)
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = de(s, i)
    end
    return Dataset{D+1, T}(data)
end

#####################################################################################
#                              MultiDimensional R                                   #
#####################################################################################
"""
    MTDelayEmbedding(D, τ, B) -> `embedding`
Return a delay coordinates embedding structure to be used as a functor,
given multiple timeseries (`B` in total), either as a [`Dataset`](@ref) or a
`SizedArray` (see [`reconstruct`](@ref)), and some index.
Calling
```julia
embedding(s, n)
```
will create the `n`-th reconstructed vector of the embedded space, which has `D`
temporal neighbors with delay(s) `τ`. See [`reconstruct`](@ref) for more.

*Be very careful when choosing `n`, because `@inbounds` is used internally.*
"""
struct MTDelayEmbedding{D, B, X} <: AbstractEmbedding
    delays::SMatrix{D, B, Int, X} # X = D*B = total dimension number
end

@inline MTDelayEmbedding(D, τ, B) = MTDelayEmbedding(Val{D}(), τ, Val{B}())
@inline function MTDelayEmbedding(::Val{D}, τ::Int, ::Val{B}) where {D, B}
    X = D*B
    idxs = SMatrix{D,B,Int,X}([k*τ for k in 1:D, j in 1:B])
    return MTDelayEmbedding{D, B, X}(idxs)
end
@inline function MTDelayEmbedding(
    ::Val{D}, τ::AbstractMatrix{<:Integer}, ::Val{B}) where {D, B}
    X = D*B
    D != size(τ)[1] && throw(ArgumentError(
    "`size(τ)[1]` must equal the number of spatial neighbors."
    ))
    B != size(τ)[2] && throw(ArgumentError(
    "`size(τ)[2]` must equal the number of timeseries."
    ))
    return MTDelayEmbedding{D, B, X}(SMatrix{D, B, Int, X}(τ))
end
function MTDelayEmbedding(
    ::Val{D}, τ::AbstractVector{<:Integer}, ::Val{B}) where {D, B}
    error("Does not work with vector τ, only matrix or integer!")
end

@generated function (r::MTDelayEmbedding{D, B, X})(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    i) where {D, A, B, T, M, X}
    gensprev = [:(s[i, $d]) for d=1:B]
    gens = [:(s[i + r.delays[$k, $d], $d]) for k=1:D for d=1:B]
    quote
        @_inline_meta
        @inbounds return SVector{$(D+1)*$B,T}($(gensprev...), $(gens...))
    end
end

@inline function reconstruct(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    D, τ) where {A, B, T, M}

    de::MTDelayEmbedding{D, B, D*B} = MTDelayEmbedding(D, τ, B)
    L = size(s)[1] - maximum(de.delays)
    X = (D+1)*B
    data = Vector{SVector{X, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = de(s, i)
    end
    return Dataset{X, T}(data)
end
