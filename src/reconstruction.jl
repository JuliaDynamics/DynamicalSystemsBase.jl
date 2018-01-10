using StaticArrays
using LsqFit: curve_fit
using StatsBase: autocor

export Reconstruction
export estimate_delay
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


#####################################################################################
#                      Estimate Reconstruction Parameters                           #
#####################################################################################
"""
    localextrema(y) -> max_ind, min_ind
Find the local extrema of given array `y`, by scanning point-by-point. Return the
indices of the maxima (`max_ind`) and the indices of the minima (`min_ind`).
"""
function localextrema end
@inbounds function localextrema(y)
    l = length(y)
    i = 1
    maxargs = Int[]
    minargs = Int[]
    if y[1] > y[2]
        push!(maxargs, 1)
    elseif y[1] < y[2]
        push!(minargs, 1)
    end

    for i in 2:l-1
        left = i-1
        right = i+1
        if  y[left] < y[i] > y[right]
            push!(maxargs, i)
        elseif y[left] > y[i] < y[right]
            push!(minargs, i)
        end
    end

    if y[l] > y[l-1]
        push!(maxargs, l)
    elseif y[l] < y[l-1]
        push!(minargs, l)
    end
    return maxargs, minargs
end


function exponential_decay_extrema(c::AbstractVector)
    ac = abs.(c)
    ma, mi = localextrema(ac)
    # ma start from 1 but correlation is expected to start from x=0
    ydat = ac[ma]; xdat = ma .- 1
    # Do curve fit from LsqFit
    model(x, p) = @. exp(-x/p[1])
    decay = curve_fit(model, xdat, ydat, [1.0]).param[1]
    return decay
end

function exponential_decay(c::AbstractVector)
    # Do curve fit from LsqFit
    model(x, p) = @. exp(-x/p[1])
    decay = curve_fit(model, 0:length(c)-1, abs.(c), [1.0]).param[1]
    return decay
end

"""
    estimate_delay(s) -> τ
Estimate an optimal delay to be used in [`Reconstruction`](@ref),
by performing an exponential fit to
the `abs.(c)` with `c` the auto-correlation function of `s`.
Return the exponential decay time `τ` rounded to an integer.
"""
function estimate_delay(x::AbstractVector)
    c = autocor(x, 0:length(x)÷10)
    i = 1
    # Find 0 crossing:
    while c[i] > 0
        i+= 1
        i == length(c) && break
    end
    # Find exponential fit:
    τ = exponential_decay(c)
    # Is there a method to deduce which one of the 2 is the better approach?
    return round(Int, τ)
end

function estimate_dimension(s::AbstractVector)
  # Estimate number of “false nearest neighbors” due to
  # projection into a too low dimension reconstruction space
end
