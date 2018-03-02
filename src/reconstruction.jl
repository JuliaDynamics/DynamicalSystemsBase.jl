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

## Description
If `τ` is an integer, then the ``n``th row of a `Reconstruction`
is
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+(D-1)\\tau))
```
If instead `τ` is a vector of integers, then the ``n``th row is
```math
(s(n+\\tau[1]), s(n+\\tau[2]), s(n+\\tau[3]), \\dots, s(n+\\tau[D]))
```

The reconstruction object `R` can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].

The case of different delay times allows reconstructing systems with many time scales,
see [3].

`R` can be accessed similarly to a [`Dataset`](@ref)
and can also be given to all functions that accept a `Dataset`
(like e.g. `generalized_dim` from module `ChaosTools`).

## Multi-dimensional `Reconstruction`
To make a reconstruction out of a multi-dimensional timeseries (i.e. trajectory) use
```julia
Reconstruction(tr::SizedAray{A, B}, D, τ)
Reconstruction(tr::AbstractDataset{B}, D, τ)
```
with `B` the "base" dimensions.

If the trajectory is for example ``(x, y)``, then the reconstruction is
```math
(x(n), y(n), x(n+\\tau), y(n+\\tau), \\dots, x(n+(D-1)\\tau), y(n+(D-1)\\tau))
```

Note that a reconstruction created
this way will have `B*D` total dimensions and *not* `D`, as a result of
each dimension of `s` having `D` delayed dimensions.

## References

[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)

[3] : K. Judd & A. Mees, Physica D **120**, pp 273 (1998)
"""
struct Reconstruction{D, T<:Number, τ} <: AbstractReconstruction{D, T, τ}
    data::Vector{SVector{D,T}}
    delay::τ
end

function Reconstruction(s::AbstractVector{T}, D, τ::DT) where {T, DT}
    if DT <: AbstractVector{Int}
        length(τ) != D && throw(ArgumentError(
        "The delay vector must have `length(τ) == D`."
        ))
    elseif DT != Int
        throw(ArgumentError(
        "Only Int or AbstractVector{Int} types are allowed for the delay."
        ))
    end
    Reconstruction{D, T, DT}(reconstruct(s, Val{D}(), τ), τ)
end

function reconstruct_impl(::Val{D}) where D
    gens = [:(s[i + $k*τ]) for k=0:D-1]

    quote
        L = length(s) - ($(D-1))*τ;
        T = eltype(s)
        data = Vector{SVector{$D, T}}(L)
        for i in 1:L
            data[i] = SVector{$D,T}($(gens...))
        end
        V = typeof(s)
        T = eltype(s)
        data
    end
end
function reconstruct_impl_tvec(::Val{D}) where D
    gens = [:(s[i + τ[$k]]) for k=1:D]

    quote
        L = length(s) - ($(D-1))*maximum(τ);
        T = eltype(s)
        data = Vector{SVector{$D, T}}(L)
        for i in 1:L
            data[i] = SVector{$D,T}($(gens...))
        end
        V = typeof(s)
        T = eltype(s)
        data
    end
end
@generated function reconstruct(s::AbstractVector{T}, ::Val{D}, τ::Int) where {D, T}
    reconstruct_impl(Val{D}())
end
@generated function reconstruct(
    s::AbstractVector{T}, ::Val{D}, τ::AbstractVector{Int}) where {D, T}
    reconstruct_impl_tvec(Val{D}())
end


# Pretty print:
matname(d::Reconstruction{D, T, τ}) where {D, T, τ} =
"(D=$(D), τ=$(d.delay)) - delay coordinates Reconstruction"
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

Reconstruction(s::AbstractDataset{B, T}, D, τ::DT) where {B, T, DT} =
MDReconstruction{B*D, D, B, T, DT}(reconstruct(s, Val{D}(), τ), τ)

Reconstruction(s::SizedArray{Tuple{A, B}, T, 2, M}, D, τ::DT) where {A, B, T, M, DT} =
MDReconstruction{B*D, D, B, T, DT}(reconstruct(s, Val{D}(), τ), τ)

# Pretty print:
matname(d::MDReconstruction{DxB, D, B, T, τ}) where {DxB, D, B, T, τ} =
"(B=$(B), D=$(D), τ=$(d.delay)) - delay coordinates multi-dimensional Reconstruction"
