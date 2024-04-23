# Printing works as follows; first a generic header is used
# then type-deduce details are printed, then dynamic rule,
# any additional information that is extendable,
# and last current parameter, time, state

Base.summary(ds::DynamicalSystem) =
"$(dimension(ds))-dimensional $(nameof(typeof(ds)))"

function Base.show(io::IO, ds::DynamicalSystem)
    print(io, summary(ds))
end

# Extend this function to return a vector of `Pair`s of `"description" => value`
additional_details(::DynamicalSystem) = []

function Base.show(io::IO, ::MIME"text/plain", ds::DynamicalSystem)
    descriptors = [
        "deterministic" => isdeterministic(ds),
        "discrete time" => isdiscretetime(ds),
        "in-place" => isinplace(ds),
        "dynamic rule" => rulestring(dynamic_rule(ds)),
    ]
    append!(descriptors, additional_details(ds))
    append!(descriptors, [
        "parameters" => current_parameters(ds),
        "time" => current_time(ds),
        "state" => current_state(ds),
    ])
    padlen = maximum(length(d[1]) for d in descriptors) + 3

    println(io, summary(ds))
    for (desc, val) in descriptors
        println(io, rpad(" $(desc): ", padlen), val)
    end

    # TODO: Improve printing of parameter and state by using `printlimited`
    # text = summary(ds)
    # u0 = get_state(ds)'
    # println(io, text)
    # prefix = rpad(" state: ", padlen)
    # print(io, prefix); printlimited(io, u0, Δx = length(prefix)); print(io, "\n")
    # println(io,  rpad(" rule f: ", padlen),     rulestring(ds))
    # println(io,  rpad(" in-place? ", padlen),   isinplace(ds))
    # println(io,  rpad(" jacobian: ", padlen),   jacobianstring(ds))
    # print(io,    rpad(" parameters: ", padlen))
    # printlimited(io, printable(ds.p), Δx = length(prefix), Δy = 10)
end

rulestring(f::Function) = nameof(f)
rulestring(f) = nameof(typeof(f))

printable(p::AbstractVector) = p'
printable(p::Nothing) = "nothing"
printable(p) = string(p)

# Credit to Sebastian Pfitzner
function printlimited(io, x; Δx = 0, Δy = 0)
    sz = displaysize(io)
    io2 = IOBuffer(); ctx = IOContext(io2, :limit => true, :compact => true,
    :displaysize => (sz[1]-Δy, sz[2]-Δx))
    if x isa AbstractArray
        Base.print_array(ctx, x)
        s = String(take!(io2))
        s = replace(s[2:end], "  " => ", ")
        Base.print(io, "["*s*"]")
    else
        Base.print(ctx, x)
        s = String(take!(io2))
        Base.print(io, s)
    end
end

printlimited(io, x::Number; kwargs...) = print(io, x)

"""
    state_name(index)::String

Return a name that matches the outcome of [`observe_state`](@ref) with `index`.
"""
state_name(f::Int) = "u"*subscript(f)
state_name(f::Function) = string(f)
state_name(f::Union{AbstractString,Symbol}) = string(f)
function state_name(f)
    # First, try a symbolic name
    try
        n = string(DynamicalSystemsBase.SymbolicIndexingInterface.getname(f))
        return n
    catch e
    end
    # if it failed, cast into string
    n = string(f)
    # and remove `(t)`
    replace(n, "(t)" => "")
end

"""
    parameter_name(index)::String

Return a name that matches the outcome of [`current_parameter`](@ref) with `index`.
"""
parameter_name(f::Int) = "p"*subscript(f)
parameter_name(f::Union{AbstractString,Symbol}) = string(f)
parameter_name(f) = string(DynamicalSystemsBase.SymbolicIndexingInterface.getname(f))

export state_name, parameter_name

"""
    subscript(i::Int)

Transform `i` to a string that has `i` as a subscript.
"""
function subscript(i::Int)
    if i < 0
        "₋"*subscript(-i)
    elseif i == 1
        "₁"
    elseif i == 2
        "₂"
    elseif i == 3
        "₃"
    elseif i == 4
        "₄"
    elseif i == 5
        "₅"
    elseif i == 6
        "₆"
    elseif i == 7
        "₇"
    elseif i == 8
        "₈"
    elseif i == 9
        "₉"
    elseif i == 0
        "₀"
    else
        join(subscript.(digits(i)))
    end
end
