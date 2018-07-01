# TODO:
# separate reconstruction to multidim and standard
# (since wehave STRec anyway)
# Clean up source
# Add tests of type stability when calling the reconstructors
# Decide on naming: I think Reconstruction should be made normal function,
# with small letters and renamed to `reconstruct`. The type Reconstruction
# still stays because I'd prefer to have the pretty printing with τ


using StaticArrays
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
    DelayEmbedding(D, τ)
Return a delay coordiantes embedding method to be used with [`reconstruct`](@ref),
having `D` temporal neighbors and delay `τ` (integer or vector).
Notice that the dimension of the reconstructed space will be `D+1`!

This embedding can be used to reconstructed one, or multiple timeseries.

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

The case of different delay times allows reconstructing systems with many time scales,
see [3].

## References

[3] : K. Judd & A. Mees, [Physica D **120**, pp 273 (1998)](https://www.sciencedirect.com/science/article/pii/S0167278997001188)
"""
struct DelayEmbedding{D}
    delays::SVector{D, Int}
end

function DelayEmbedding(D, τ)
    if typeof(τ) <: Integer
        idxs = [k*τ for k in 1:D]
        return DelayEmbedding{D}(SVector{D, Int}(idxs...))
    elseif typeof(τ) <: AbstractArray{<:Integer}
        D != length(τ) && throw(ArgumentError(
        "Delay time vector length must equal the number of spatial neighbors."
        ))
        if !issorted(τ)
            @warn "Delay times are not sorted. Sorting now."
            τ = sort(τ)
        end
        return DelayEmbedding{D}(SVector{D, Int}(τ...))
    end
end

@generated function (r::DelayEmbedding{D})(s::AbstractArray{T}, i) where {D, T}
    gens = [:(s[i + r.delays[$k]]) for k=1:D]
    quote
        @inbounds return SVector{$D,T}($(gens...))
    end
end

"""
    reconstruct(s, de::AbstractEmbedding)
Reconstruct `s` using the embedding `de` (any subtype of `AbstractEmbedding`,
like e.g. [`DelayEmbedding`](@ref)).
Return the result as a [`Dataset`](@ref). See the documentations of the individual
embedding methods for what kind of data can be reconstructed.

The reconstructed dataset can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].

## References
[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)
"""
@inline function reconstruct(s::AbstractVector{T}, de::DelayEmbedding{D}) where {T, D}
    L = length(s) - D*de.delays[end]
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = de(s, i)
    end
    return Dataset{D+1, T}(data)
end

#####################################################################################
#                              MultiDimensional R                                   #
#####################################################################################
# TODO:
struct MTDelayEmbedding <: AbstractEmbedding
    delays::Matrix{Int}
end


struct MDReconstruction{DxB, D, B, T<:Number, τ} <: AbstractReconstruction{DxB, T, τ}
    data::Vector{SVector{DxB,T}}
    delay::τ
end

"""
## Multi-dimensional `Reconstruction`
To make a reconstruction out of a multi-dimensional timeseries (i.e. trajectory) use
```julia
Reconstruction(tr::SizedAray{A, B}, D, τ)
Reconstruction(tr::AbstractDataset{B}, D, τ)
```
with `B` the "base" dimensions.

If the trajectory is for example ``(x, y)``, then the ``n``th row is
```math
(x(n), y(n), x(n+\\tau), y(n+\\tau), \\dots, x(n+(D-1)\\tau), y(n+(D-1)\\tau))
```
for integer `τ` and if `τ` is an `AbstractMatrix{Int}`, so that `size(τ) == (D, B)`,
then the ``n``th row is
```math
(x(n+\\tau[1, 1]), y(n+\\tau[1, 2]), \\dots, x(n+\\tau[D, 1]), y(n+\\tau[D, 2]))
```

Note that a reconstruction created
this way will have `B*D` total dimensions and *not* `D`, as a result of
each dimension of `s` having `D` delayed dimensions.
"""

struct MDDelayVector{D, DxB, TAU}
    τ::TAU
end

@generated function (r::MDDelayVector{D, Int})(
    s::AbstractDataset{B, T}, i)

    gens = [:(s[i + $k*τ, $d]) for k=0:D-1 for d=1:B]
    quote
        @inbounds return SVector{$D*B,T}($(gens...))
    end
end



@generated function (r::DelayVector{D, Vector{<:Integer}})(
    s::SizedArray{Tuple{A, B}, T, 2, M}, i) where {D, A, B, T, M}

    gens = [:(s[i + τ[$k, $d], $d]) for k=1:D for d=1:B]
    quote
        @inbounds SVector{$D,T}($(gens...))
    end
end



function reconstructmat_impl(::Val{S2}, ::Val{D}) where {S2, D}
    gens = [:(s[i + $k*τ, $d]) for k=0:D-1 for d=1:S2]

    quote
        L = size(s,1) - ($(D-1))*τ;
        T = eltype(s)
        data = Vector{SVector{$D*$S2, T}}(L)
        for i in 1:L
            data[i] = SVector{$D*$S2,T}($(gens...))
        end
        V = typeof(s)
        T = eltype(s)
        data
    end
end

@generated function reconstruct(s::SizedArray{Tuple{A, B}, T, 2, M}, ::Val{D}, τ) where {A, B, T, M, D}
    reconstructmat_impl(Val{B}(), Val{D}())
end
@generated function reconstruct(s::AbstractDataset{B, T}, ::Val{D}, τ) where {B, T, D}
    reconstructmat_impl(Val{B}(), Val{D}())
end

Reconstruction(s::AbstractDataset{B, T}, D, τ::Int) where {B, T} =
MDReconstruction{B*D, D, B, T, Int}(reconstruct(s, Val{D}(), τ), τ)

Reconstruction(s::SizedArray{Tuple{A, B}, T, 2, M}, D, τ::Int) where {A, B, T, M} =
MDReconstruction{B*D, D, B, T, Int}(reconstruct(s, Val{D}(), τ), τ)

## Multi-time version
function reconstructmat_impl_tvec(::Val{S2}, ::Val{D}) where {S2, D}
    gens = [:(s[i + τ[$k, $d], $d]) for k=1:D for d=1:S2]

    quote
        L = size(s,1) - maximum(τ);
        T = eltype(s)
        data = Vector{SVector{$D*$S2, T}}(L)
        for i in 1:L
            data[i] = SVector{$D*$S2,T}($(gens...))
        end
        V = typeof(s)
        T = eltype(s)
        data
    end
end

@generated function reconstruct_multi(s::SizedArray{Tuple{A, B}, T, 2, M}, ::Val{D}, τ) where {A, B, T, M, D}
    reconstructmat_impl_tvec(Val{B}(), Val{D}())
end
@generated function reconstruct_multi(s::AbstractDataset{B, T}, ::Val{D}, τ) where {B, T, D}
    reconstructmat_impl_tvec(Val{B}(), Val{D}())
end

function Reconstruction(
    s::AbstractDataset{B, T}, D, τ::DT) where {B, T, DT<:AbstractMatrix{Int}}
    size(τ) != (D, B) && throw(ArgumentError(
    "The delay matrix must have `size(τ) == (D, B)`."
    ))
    return MDReconstruction{B*D, D, B, T, DT}(reconstruct_multi(s, Val{D}(), τ), τ)
end

function Reconstruction(
    s::SizedArray{Tuple{A, B}, T, 2, M}, D, τ::DT
    ) where {A, B, T, M, DT<:AbstractMatrix{Int}}
    size(τ) != (D, B) && throw(ArgumentError(
    "The delay matrix must have `size(τ) == (D, B)`."
    ))
    return MDReconstruction{B*D, D, B, T, DT}(reconstruct_multi(s, Val{D}(), τ), τ)
end

# Pretty print:
Base.summary(d::MDReconstruction{DxB, D, B, T, τ}) where {DxB, D, B, T, τ} =
"(B=$(B), D=$(D), τ=$(d.delay)) - delay coordinates multi-dimensional Reconstruction"
