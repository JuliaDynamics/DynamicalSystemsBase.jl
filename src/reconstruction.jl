# TODO:
# separate reconstruction to multidim and standard
# (since wehave STRec anyway)
# Clean up source
# Add tests of type stability when calling the reconstructors
# Decide on naming: I think Reconstruction should be made normal function,
# with small letters and renamed to `reconstruct`. The type Reconstruction
# still stays because I'd prefer to have the pretty printing with τ


using StaticArrays
export Reconstruction, MDReconstruction

#####################################################################################
#                            Reconstruction Object                                  #
#####################################################################################
abstract type AbstractReconstruction{D, T, τ} <: AbstractDataset{D, T} end

"""
    Reconstruction(s::AbstractVector, D, τ) <: AbstractDataset
`D`-dimensional delay-coordinates reconstruction object with delay `τ`,
created from a timeseries `s`.

Notice that the number of temporal neighbors is `D-1`!

## Description
If `τ` is an integer, then the ``n``th row of a `Reconstruction`
is
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+(D-1)\\tau))
```
If instead `τ` is a vector of integers, so that `length(τ) == D`,
then the ``n``th row is
```math
(s(n+\\tau[1]), s(n+\\tau[2]), s(n+\\tau[3]), \\dots, s(n+\\tau[D]))
```

The reconstruction object `R` can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].

The case of different delay times allows reconstructing systems with many time scales,
see [3].

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

## References

[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)

[3] : K. Judd & A. Mees, [Physica D **120**, pp 273 (1998)](https://www.sciencedirect.com/science/article/pii/S0167278997001188)
"""
struct Reconstruction{D, T<:Number, TAU}# <: AbstractReconstruction{D, T, τ}
    data::Vector{SVector{D,T}}
    τ::TAU
end

Base.summary(d::Reconstruction{D, T}) where {D, T} =
"(D=$(D), τ=$(d.τ)) - delay coordinates Reconstruction"

struct Reconstructor{D, TAU}
    τ::TAU
end

@generated function (r::Reconstructor{D, Int})(s::AbstractArray{T}, i) where {D, T}
    gens = [:(Base.unsafe_getindex(s, i + $k*r.τ)) for k=0:D-1]
    quote
        @inbounds return SVector{$D,T}($(gens...))
    end
end
@generated function (r::Reconstructor{D, Vector{<:Integer}})(s::AbstractArray{T}, i) where {D, T}
    gens = [:(Base.unsafe_getindex(s, i + r.τ[$k])) for k=1:D]
    quote
        SVector{$D,T}($(gens...))
    end
end

reconstructed_length(s, D, τ::Integer) = length(s) - ((D-1))*τ;
reconstructed_length(s, D, τ::AbstractVector{<:Integer}) = length(s) - ((D-1))*maximum(τ);

@inline function delaycoordinates(s::AbstractVector{T}, D::Int, τ::DT) where {T, DT}
    L = reconstructed_length(s, D, τ)
    r = Reconstructor{D,DT}(τ)
    data = Vector{SVector{D, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = r(s, i)
    end
    return Reconstruction{D, T, DT}(data, τ)
end





#####################################################################################
#                              MultiDimensional R                                   #
#####################################################################################
struct MDReconstruction{DxB, D, B, T<:Number, τ} <: AbstractReconstruction{DxB, T, τ}
    data::Vector{SVector{DxB,T}}
    delay::τ
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
