# Printing works as follows; first a generic header is used
# then type-deduce details are printed, then dynamic rule,
# any additional information that is extendable,
# and last current parameter, time, state

Base.summary(ds::DynamicalSystem) =
"$(dimension(ds))-dimensional $(nameof(typeof(ds)))"

# Extend this function to return a vector of `Pair`s of `"description" => value`
additional_details(::DynamicalSystem) = []

function Base.show(io::IO, ds::DynamicalSystem)
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