using StaticArrays
using LsqFit: curve_fit
using StatsBase: autocor

export Reconstruction
#####################################################################################
#                            Reconstruction Object                                  #
#####################################################################################
"""
    Reconstruction{D, T, τ} <: AbstractDataset{D, T}
`D`-dimensional delay-coordinates reconstruction object with delay `τ`,
created from a timeseries `s` with `T` type numbers.

Use `Reconstruction(s::AbstractVector{T}, D, τ)` to create an instance.

## Description
The ``n``th row of a `Reconstruction` is the `D`-dimensional vector
```math
(s(n), s(n+\\tau), s(n+2\\tau), \\dots, s(n+(D-1)\\tau))
```

The reconstruction object `R` can have same
invariant quantities (like e.g. lyapunov exponents) with the original system
that the timeseries were recorded from, for proper `D` and `τ` [1, 2].

`R` can be accessed similarly to a [`Dataset`](@ref):
```julia
s = rand(1e6)
R = Reconstruction(s, 4, 1) # dimension 4 and delay 1
R[3] # third point of reconstruction, ≡ (s[3], s[4], s[5], s[6])
R[1, 2] # Second element of first point of reconstruction, ≡ s[2]
```
and can also be given to all functions that accept a `Dataset`
(like e.g. `generalized_dim` from module `ChaosTools`).

The functions `dimension(R)` and `delay(R)` return `D` and `τ` respectively.

## References

[1] : F. Takens, *Detecting Strange Attractors in Turbulence — Dynamical
Systems and Turbulence*, Lecture Notes in Mathematics **366**, Springer (1981)

[2] : T. Sauer *et al.*, J. Stat. Phys. **65**, pp 579 (1991)
"""
type Reconstruction{D, T<:Number, τ} <: AbstractDataset{D, T}
    data::Vector{SVector{D,T}}
end

Reconstruction(s::AbstractVector{T}, D, τ) where {T} =
Reconstruction{D, T, τ}(reconstruct(s, Val{D}(), τ))

@inline delay(::Reconstruction{D, T, t}) where {T,D,t} = t

function reconstruct_impl(::Type{Val{D}}) where D
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
    reconstruct_impl(Val{D})
end


# Pretty print:
matname(d::Reconstruction{D, T, τ}) where {D, T, τ} =
"(D=$(D), τ=$(τ)) - delay coordinates Reconstruction"
