using StaticArrays
export Reconstruction, MDReconstruction
export estimate_delay
export estimate_dimension

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
    s::AbstractVector{T}, ::Val{D}, τ::AbstractArray{Int}) where {D, T}
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
matname(d::MDReconstruction{DxB, D, B, T, τ}) where {DxB, D, B, T, τ} =
"(B=$(B), D=$(D), τ=$(d.delay)) - delay coordinates multi-dimensional Reconstruction"



#####################################################################################
#                               Estimate Delay Times                                #
#####################################################################################
using NearestNeighbors
using Distances: chebyshev
using SpecialFunctions: digamma
using LsqFit: curve_fit
using StatsBase: autocor

export estimate_delay, mutual_info

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

function mutual_info(Xm::Vararg{<:AbstractVector,M}; k=1) where M
    @assert M > 1
    @assert (size.(Xm,1) .== size(Xm[1],1)) |> prod
    N = size(Xm[1],1)

    d = Dataset(Xm...)
    tree = KDTree(d.data, Chebyshev()) # replacing this with Euclidean() gives a
                                       # speed-up of x4

    n_x_m = zeros(M)

    Xm_sp = zeros(Int, N, M)
    Xm_revsp = zeros(Int, N, M)
    for m in 1:M #this loop takes 5% of time
        Xm_sp[:,m] .= sortperm(Xm[m]; alg=QuickSort)
        Xm_revsp[:,m] .= sortperm(sortperm(Xm[m]; alg=QuickSort); alg=QuickSort)
    end

    I = digamma(k) - (M-1)/k + (M-1)*digamma(N)

    I_itr = zeros(M)
    # Makes more sense computationally to loop over N rather than M
    for i in 1:N
        point = d[i]
        idxs, dists = knn(tree, point, k+1)  # this line takes 80% of time

        ϵi = idxs[indmax(dists)]
        ϵ = abs.(d[ϵi] - point)  # surprisingly, this takes a significant amount of time

        for m in 1:M # this loop takes 8% of time
            hb = lb = Xm_revsp[i,m]
            while abs(Xm[m][Xm_sp[hb,m]] - Xm[m][i]) <= ϵ[m] && hb < N
                hb += 1
            end
            while abs(Xm[m][Xm_sp[lb,m]] - Xm[m][i]) <= ϵ[m] && lb > 1
                lb -= 1
            end
            n_x_m[m] = hb - lb
        end

        I_itr .+= digamma.(n_x_m)
    end

    I_itr ./= N

    I -= sum(I_itr)

    return max(0, I)
end


"""
    estimate_delay(s, method::String) -> τ

Estimate an optimal delay to be used in [`Reconstruction`](@ref).

The `method` can be one of the following:

* `first_zero` : find first delay at which the auto-correlation function becomes 0.
* `first_min` : return delay of first minimum of the auto-correlation function.
* `exp_decay` : perform an exponential fit to the `abs.(c)` with `c` the auto-correlation function of `s`.
  Return the exponential decay time `τ` rounded to an integer.
"""
function estimate_delay(x::AbstractVector, method::String; maxtau=100, k=1)
    method ∈ ["first_zero", "first_min", "exp_decay", "mutual_inf"] || throw(ArgumentError("Unknown method"))
     if method=="first_zero"
        c = autocor(x, 0:length(x)÷10; demean=true)
        i = 1
        # Find 0 crossing:
        while c[i] > 0
            i+= 1
            i == length(c) && break
        end
        return i

    elseif method=="first_min"
        c = autocor(x, 0:length(x)÷10, demean=true)
        i = 1
        # Find min crossing:
        while  c[i+1] < c[i]
            i+= 1
            i == length(c)-1 && break
        end
        return i
    elseif method=="exp_decay"
        c = autocor(x, 0:length(x)÷10, demean=true)
        # Find exponential fit:
        τ = exponential_decay(c)
        return round(Int,τ)
    #Need a package that can be precompiled...
    elseif method=="mutual_inf"
        m = mutual_info(x,x; k=k)
        for i=1:maxtau
            n = mutual_info(x[1:end-i],x[1+i:end]; k=k)
            n > m && return i
            m = n
        end
    end
end


#####################################################################################
#                                Estimate Dimension                                 #
#####################################################################################



function _average_a(s::AbstractVector{T},D,τ) where T
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    R1 = Reconstruction(s,D+1,τ)
    R2 = Reconstruction(s[1:end-τ],D,τ)
    tree2 = KDTree(R2)
    nind = (x = knn(tree2, R2.data, 2)[1]; [ind[1] for ind in x])
    e=0.
    for (i,j) in enumerate(nind)
        δ = norm(R2[i]-R2[j], Inf)
        #If R2[i] and R2[j] are still identical, choose the next nearest neighbor
        if δ == 0.
            j = knn(tree2, R2[i], 3, true)[1][end]
            δ = norm(R2[i]-R2[j], Inf)
        end
        e += norm(R1[i]-R1[j], Inf) / δ
    end
    return e / length(R1)
end

function dimension_indicator(s,D,τ) #this is E1, equation (3) of Cao
    return _average_a(s,D+1,τ)/_average_a(s,D,τ)
end


"""
    estimate_dimension(s::AbstractVector, τ:Int, Ds) -> E1s

Estimate an optimal embedding dimension to be used in [`Reconstruction`](@ref).

## Description
Take a scalar timeseries `s`, the embedding delay `τ` and a the dimensions `Ds` to check,
and compute the paramter `E1` for each dimension according to Cao's Method [1].

Return a vector of all computed `E1`s. To estimate a dimension from this,
find the dimension for which the value `E1` saturates to 1 (see
`saturation_point` from `ChaosTools`).

*Note: This method does not work for datasets with perfectly periodic signals.*


## References

[1] : Liangyue Cao, *Practical method for determining the minimum embedding dimension
of a scalar time series*, Physica D, pp. 43-50 (1997)
"""
function estimate_dimension(s::AbstractVector{T}, τ::Int, Ds = 1:6) where {T}
    E1s = zeros(T, length(Ds))
    aafter = zero(T)
    aprev = _average_a(s, Ds[1], τ)
    for (i, D) ∈ enumerate(Ds)
        aafter = _average_a(s, D+1, τ)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end

# then use function `saturation_point(Ds, E1s)` from ChaosTools

function stochastic_indicator(s::AbstractVector{T},D,τ) where T # E2, equation (5)
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    #Calculate E* for Dimension D+1
    R1 = Reconstruction(s,D+1,τ)
    tree1 = KDTree(R1[1:end-1-τ])
    method = FixedMassNeighborhood(2)

    Es1 = 0.
    nind = (x = neighborhood(R1[1:end-τ], tree1, method); [ind[1] for ind in x])
    for  (i,j) ∈ enumerate(nind)
        Es1 += abs(R1[i+τ][end] - R1[j+τ][end]) / length(R1)
    end

    #Calculate E* for Dimension D
    R2 = Reconstruction(s,D,τ)
    tree2 = KDTree(R2[1:end-1-τ])
    Es2 = 0.
    nind = (x = neighborhood(R2[1:end-τ], tree2, method); [ind[1] for ind in x])
    for  (i,j) ∈ enumerate(nind)
        Es2 += abs(R2[i+τ][end] - R2[j+τ][end]) / length(R2)
    end
    return Es1/Es2
end
