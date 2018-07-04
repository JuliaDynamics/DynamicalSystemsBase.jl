export reconstruct, DelayEmbedding, AbstractEmbedding

#####################################################################################
#                            Reconstruction Object                                  #
#####################################################################################
"""
    AbstractEmbedding
Super-type of embedding methods. Use `subtypes(AbstractEmbedding)` for available
methods.
"""
abstract type AbstractEmbedding end

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
    # Notice that the type-parameter D here is not the number of temporal neighbors
    # but plus one. That is because you cannot declare something like
    # ::SVector{D+1}. It is not allowed.
    delays::SVector{D, Int}
end

function DelayEmbedding(D, τ)
    if typeof(τ) <: Integer
        idxs = [k*τ for k in 0:D]
        return DelayEmbedding{D+1}(SVector{D+1, Int}(idxs...))
    elseif typeof(τ) <: AbstractArray{<:Integer}
        D != length(τ) && throw(ArgumentError(
        "Delay time vector length must equal the number of spatial neighbors."
        ))
        if !issorted(τ)
            @warn "Delay times are not sorted. Sorting now."
            τ = sort(τ)
        end
        return DelayEmbedding{D+1}(SVector{D+1, Int}(0, τ...))
    end
end

@generated function (r::DelayEmbedding{D})(s::AbstractArray{T}, i) where {D, T}
    gens = [:(s[i + r.delays[$k]]) for k=1:D]
    quote
        @inbounds return SVector{$D,T}($(gens...))
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
    de = DelayEmbedding(D, τ)
    L = length(s) - de.delays[end]
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
    # Again, here D is the number of temporal neighbors *plus one*.
    delays::SMatrix{D, B, Int, X} # X = D*B = total dimension number
end

function MTDelayEmbedding(D, τ, B)
    X = (D+1)*B
    if typeof(τ) <: Integer
        idxs = SMatrix{D+1,B,Int,X}([k*τ for k in 0:D, j in 1:B])
        return MTDelayEmbedding{D+1, B, X}(idxs)
    elseif typeof(τ) <: AbstractMatrix{<:Integer}
        D != size(τ)[1] && throw(ArgumentError(
        "`size(τ)[1]` must equal the number of spatial neighbors."
        ))
        return MTDelayEmbedding{D+1, B, X}(SMatrix{D+1, B, Int, X}(zeros(B)..., τ...))
    else
	return ArgumentError("Please make sure τ is a Matrix")
    end
end

@generated function (r::MTDelayEmbedding{D, B, X})(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    i) where {D, A, B, T, M, X}

    gens = [:(s[i + r.delays[$k, $d], $d]) for k=1:D for d=1:B]

    quote
        @inbounds return SVector{$D*$B,T}($(gens...))
    end
end

function reconstruct(
    s::Union{AbstractDataset{B, T}, SizedArray{Tuple{A, B}, T, 2, M}},
    D, τ) where {A, B, T, M}

    de = MTDelayEmbedding(D, τ, B)
    L = length(s) - maximum(de.delays)
    X = (D+1)*B
    data = Vector{SVector{X, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = de(s, i)
    end
    return Dataset{B*(D+1), T}(data)
end
