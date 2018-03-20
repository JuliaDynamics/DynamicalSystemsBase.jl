using StaticArrays, Requires
using IterTools: chain
import Base: ==

export Dataset, AbstractDataset, minima, maxima
export minmaxima, columns

abstract type AbstractDataset{D, T} end

# Size:
@inline Base.length(d::AbstractDataset) = length(d.data)
@inline Base.size(d::AbstractDataset{D,T}) where {D,T} = (length(d.data), D)
@inline Base.size(d::AbstractDataset, i::Int) = size(d)[i]
@inline Base.iteratorsize(d::AbstractDataset) = Base.HasLength()

# 1D indexing  over the container elements:
@inline Base.getindex(d::AbstractDataset, i) = d.data[i]
@inline Base.endof(d::AbstractDataset) = endof(d.data)
# 2D indexing exactly like if the dataset was a matrix
# with each column a dynamic variable
@inline Base.getindex(d::AbstractDataset, i::Int, j::Int) = d.data[i][j]
@inline Base.getindex(d::AbstractDataset, i::Colon, j::Int) =
[d.data[k][j] for k in 1:length(d)]
@inline Base.getindex(d::AbstractDataset, i::Int, j::Colon) = d.data[i]
@inline Base.getindex(d::AbstractDataset, r::Range) = d.data[r]
# Indexing with ranges
@inline Base.getindex(d::AbstractDataset, i::Range, j::Int) =
[d.data[k][j] for k in i]
@inline Base.getindex(d::AbstractDataset, i::Int, j::Range) =
[d.data[i][k] for k in j]

function Base.getindex(d::AbstractDataset{D,T}, i::AbstractVector{Int},
    j::AbstractVector{Int}) where {D,T}
    I = length(i)
    J = length(j)
    ret = zeros(T, J,I)
    for k=1:I
        for l=1:J
            ret[l,k] = d[i[k],j[l]]
        end
    end
    return reinterpret(Dataset, ret)
end

# This function should be re-enabled in Julia 0.7
# function mygetindex(d::AbstractDataset{D,T}, I::AbstractVector{Int},
#     J::AbstractVector{Int}) where {D,T}
#
#     L = length(J)
#     sind::SVector{L, Int} = SVector{L, Int}(J)
#
#     return Base.getindex(d, I, sind)
# end
function Base.getindex(d::AbstractDataset{D, T}, I::AbstractVector{Int},
    sind::SVector{L, Int}) where {D, T, L}
    ret::Vector{SVector{L, T}} = Vector{SVector{L, T}}(length(I))
    i = 1
    for k ∈ I
        ret[i] = d[k][sind]
        i += 1
    end
    return Dataset{L, T}(ret)
end

function Base.getindex(d::AbstractDataset{D,T},
    ::Colon, j::AbstractVector{Int}) where {D, T}
    return Base.getindex(d, 1:length(d), j)
end

"""
    columns(dataset) -> x, y, z, ...
Return the individual columns of the dataset.
"""
function columns end
@generated function columns(data::AbstractDataset{D, T}) where {D, T}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end

# Itereting interface:
@inline Base.eachindex(D::AbstractDataset) = Base.OneTo(length(D.data))
@inline Base.start(d::AbstractDataset) = 1
@inline Base.next(d::AbstractDataset, state) = (d[state], state + 1)
@inline Base.done(d::AbstractDataset, state) = state ≥ length(d.data) + 1

# Other commonly used functions:
Base.append!(d1::AbstractDataset, d2::AbstractDataset) = append!(d1.data, d2.data)
Base.push!(d::AbstractDataset, new_item) = push!(d.data, new_item)
@inline dimension(::AbstractDataset{D,T}) where {D,T} = D
@inline Base.eltype(d::AbstractDataset{D,T}) where {D,T} = T
==(d1::AbstractDataset, d2::AbstractDataset) = d1.data == d2.data

"""
    Dataset{D, T} <: AbstractDataset{D,T}
A dedicated interface for datasets, i.e. vectors of vectors.
It contains *equally-sized datapoints* of length `D`,
represented by `SVector{D, T}`.

It can be used exactly like a matrix that has each of the columns be the
timeseries of each of the dynamic variables. [`trajectory`](@ref) always returns
a `Dataset`. For example,
```julia
ds = Systems.towel()
data = trajectory(ds, 1000) #this returns a dataset
data[:, 2] # this is the second variable timeseries
data[1] == data[1, :] # this is the first datapoint (D-dimensional)
data[5, 3] # value of the third variable, at the 5th timepoint
```

Use `Matrix(dataset)` or `reinterpret(Matrix, dataset)` and
`Dataset(matrix)` or `reinterpret(Dataset, matrix)` to convert. The `reinterpret`
methods are cheaper but assume that each variable/timeseries is a *row* and not
column of the `matrix`.

If you have various timeseries vectors `x, y, z, ...` pass them like
`Dataset(x, y, z, ...)`. You can use `columns(dataset)` to obtain the reverse,
i.e. all columns of the dataset in a tuple.
"""
struct Dataset{D, T<:Number} <: AbstractDataset{D,T}
    data::Vector{SVector{D,T}}
end
# Empty dataset:
Dataset{D, T}() where {D,T} = Dataset(SVector{D,T}[])

###########################################################################
# Dataset(Vectors of stuff)
###########################################################################
function Dataset(v::Vector{<:AbstractArray{T}}) where {T<:Number}
    D = length(v[1])
    L = length(v)
    data = Vector{SVector{D, T}}(L)
    for i in 1:length(v)
        D != length(v[i]) && throw(ArgumentError(
        "All data-points in a Dataset must have same size"
        ))
        @inbounds data[i] = SVector{D,T}(v[i])
    end
    return Dataset{D, T}(data)
end

@generated function _dataset(::Val{D}, vecs::Vararg{<:AbstractVector{T}}) where {D, T}
    gens = [:(vecs[$k][i]) for k=1:D]

    quote
        L = length(vecs[1])
        data = Vector{SVector{$D, T}}(L)
        for i in 1:L
            data[i] = SVector{$D, T}($(gens...))
        end
        data
    end
end

function Dataset(vecs::Vararg{<:AbstractVector{T}}) where {T}
    D = length(vecs)
    return Dataset(_dataset(Val{D}(), vecs...))
end


#####################################################################################
#                                Dataset <-> Matrix                                 #
#####################################################################################
#### From dataset to matrix ####
function Base.convert(::Type{Matrix{S}}, d::AbstractDataset{D,T}) where {S, D, T}
    mat = Matrix{S}(length(d), D)
    for j in 1:D
        for i in 1:length(d)
            @inbounds mat[i,j] = d.data[i][j]
        end
    end
    mat
end
Base.convert(::Type{Matrix}, d::AbstractDataset{D,T}) where {D, T} =
convert(Matrix{T}, d)

function Base.reinterpret(::Type{M}, d::AbstractDataset{D,T}) where {M<:Matrix, D, T}
    L = length(d)
    reinterpret(T, d.data, (D,L))
end

#### From matrix to dataset ####
function Base.convert(::Type{Dataset}, mat::AbstractMatrix)
    m = transpose(mat)
    reinterpret(Dataset, m)
end

function Base.reinterpret(::Type{Dataset}, mat::Array{T,2}) where {T<:Real}
    s = size(mat)
    D = s[1]; N = s[2]
    Dataset(reinterpret(SVector{D, T}, mat, (N,)))
end

function Base.convert(::Type{Dataset}, y::Vector{T}) where {T}
    data = reinterpret(SVector{1,T}, y, (length(y),))
    return Dataset(data)
end

#####################################################################################
#                                   Pretty Printing                                 #
#####################################################################################
function matname(d::Dataset{D, T}) where {D, T}
    N = length(d)
    return "$D-dimensional Dataset{$(T)} with $N points:"
end

function matstring(d::AbstractDataset{D, T}) where {D, T}
    N = length(d)
    if N > 50
        mat = zeros(eltype(d), 50, D)
        for (i, a) in enumerate(chain(1:25, N-24:N))
            mat[i, :] .= d[a]
        end
    else
        mat = Matrix(d)
    end
    s = sprint(io -> show(IOContext(io, limit=true), MIME"text/plain"(), mat))
    s = join(split(s, '\n')[2:end], '\n')
    tos = matname(d)*"\n"*s
    return tos
end

@require Juno begin
    function Juno.render(i::Juno.Inline, d::AbstractDataset)
    tos = matstring(d)
    Juno.render(i, Juno.Tree(Text(tos), []))
    end
end

Base.show(io::IO, d::AbstractDataset) = println(io, matstring(d))


#####################################################################################
#                                 Minima and Maxima                                 #
#####################################################################################
"""
    minima(dataset)
Return an `SVector` that contains the minimum elements of each timeseries of the
dataset.
"""
function minima(data::AbstractDataset{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] < m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D,T}(m)
end

"""
    maxima(dataset)
Return an `SVector` that contains the maximum elements of each timeseries of the
dataset.
"""
function maxima(data::AbstractDataset{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D, T}(m)
end

"""
    minmaxima(dataset)
Return `minima(dataset), maxima(dataset)` without doing the computation twice.
"""
function minmaxima(data::AbstractDataset{D, T}) where {D, T<:Real}
    mi = Vector(data[1])
    ma = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > ma[i]
                ma[i] = point[i]
            elseif point[i] < mi[i]
                mi[i] = point[i]
            end
        end
    end
    return SVector{D, T}(mi), SVector{D, T}(ma)
end

#####################################################################################
#                                     SVD                                           #
#####################################################################################
# SVD of Base seems to be much faster when the "long" dimension of the matrix
# is the first one, probably due to Julia's column major structure.
# This does not depend on using `svd` or `svdfact`, both give same timings.
# In fact it is so much faster, that it is *much* more worth it to
# use `Matrix(data)` instead of `reinterpret` in order to preserve the
# long dimension being the first.
"""
    svd(d::AbstractDataset) -> U, S, Vtr
Perform singular value decomposition on the dataset.
"""
function Base.svd(d::AbstractDataset)
    F = svdfact(Matrix(d))
    return F[:U], F[:S], F[:Vt]
end
#####################################################################################
#                                    Dataset IO                                     #
#####################################################################################
"""
    read_dataset(file, ::Type{<:Dataset}, delim::Char = '\t'; skipstart = 0)
Read a `delim`-delimited text file directly into a dataset of dimension `D`
with numbers of type `T`.

Optionally skip the first `skipstart` rows of the file (that may e.g.
contain headers).

Call like `read_dataset("file.txt", Dataset{3, Float64})`.
"""
function read_dataset(filename, ::Type{Dataset{D, T}}, delim::Char = '\t';
    skipstart = 0) where {D, T}

    V = SVector{D, T}
    data = SVector{D, T}[]
    open(filename) do io
        for (i, ss) in enumerate(eachline(io))
            i ≤ skipstart && continue
            s = split(ss, delim)
            push!(data, V(ntuple(k -> parse(T, s[k]), Val(D))))
        end
    end
    return Dataset(data)
end

"""
    write_dataset(file, dataset::AbstractDataset, delim::Char = '\t'; opts...)
Write a `dataset` in a `delim`-delimited text file.

`opts` are keyword arguments passed into `writedlm`.
"""
write_dataset(f, dataset::AbstractDataset, delim::Char = '\t'; opts...) =
writedlm(f, dataset.data, delim; opts...)
