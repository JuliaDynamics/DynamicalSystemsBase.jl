Base.summary(ds::DS) =
"$(dimension(ds))-dimensional "*systemtype(ds)*" dynamical system"

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

function Base.show(io::IO, ds::DS)
    ps = 14
    text = summary(ds)
    u0 = get_state(ds)'
    println(io, text)
    prefix = rpad(" state: ", ps)
    print(io, prefix); printlimited(io, u0, Δx = length(prefix)); print(io, "\n")
    println(io,  rpad(" rule f: ", ps),     get_rule_for_print(ds))
    println(io,  rpad(" in-place? ", ps),   isinplace(ds))
    println(io,  rpad(" jacobian: ", ps),   jacobianstring(ds))
    print(io,    rpad(" parameters: ", ps))
    printlimited(io, printable(ds.p), Δx = length(prefix), Δy = 10)
end

printable(p::AbstractVector) = p'
printable(p::Nothing) = "nothing"
printable(p) = p

get_rule_for_print(a::SciMLBase.DEIntegrator) = eomstring(a.f.f)
get_rule_for_print(a::MinimalDiscreteIntegrator) = eomstring(a.f)
get_rule_for_print(a::DynamicalSystem) = eomstring(a.f)
get_rule_for_print(a) = get_rule_for_print(a.integ)

eomstring(f::Function) = nameof(f)
eomstring(f) = nameof(typeof(f))

jacobianstring(ds::DS) = isautodiff(ds) ? "ForwardDiff" : "$(eomstring(ds.jacobian))"
