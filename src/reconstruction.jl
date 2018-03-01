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
In the case of reconstrucing a timeseries, the ``n``th row of a `Reconstruction`
is the `D`-dimensional vector
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+(D-1)\\tau))
```

The reconstruction object `R` can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].

`R` can be accessed similarly to a [`Dataset`](@ref)
and can also be given to all functions that accept a `Dataset`
(like e.g. `generalized_dim` from module `ChaosTools`).

Use `delay(R)` to get `τ`.

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
"""
struct Reconstruction{D, T<:Number, τ} <: AbstractReconstruction{D, T, τ}
    data::Vector{SVector{D,T}}
end

@inline delay(::Reconstruction{D, T, t}) where {T,D,t} = t

Reconstruction(s::AbstractVector{T}, D, τ) where {T} =
Reconstruction{D, T, τ}(reconstruct(s, Val{D}(), τ))

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
@generated function reconstruct(s::AbstractVector{T}, ::Val{D}, τ) where {D, T}
    reconstruct_impl(Val{D}())
end


# Pretty print:
matname(d::Reconstruction{D, T, τ}) where {D, T, τ} =
"(D=$(D), τ=$(τ)) - delay coordinates Reconstruction"
#####################################################################################
#                              MultiDimensional R                                   #
#####################################################################################
struct MDReconstruction{DxB, D, B, T<:Number, τ} <: AbstractReconstruction{DxB, T, τ}
    data::Vector{SVector{DxB,T}}
end

Reconstruction(s::AbstractDataset{B, T}, D, τ) where {B, T} =
MDReconstruction{B*D, D, B, T, τ}(reconstruct(s, Val{D}(), τ))

Reconstruction(s::SizedArray{Tuple{A, B}, T, 2, M}, D, τ) where {A, B, T, M} =
MDReconstruction{B*D, D, B, T, τ}(reconstruct(s, Val{D}(), τ))


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


# Pretty print:
matname(d::MDReconstruction{DxB, D, B, T, τ}) where {DxB, D, B, T, τ} =
"(B=$(B), D=$(D), τ=$(τ)) - delay coordinates multi-dimensional Reconstruction"
