using DynamicalSystemsBase
using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "DynamicalSystemsBase" begin
    testfile("discrete.jl")
    testfile("continuous.jl")
    testfile("stroboscopic.jl")
    testfile("parallel.jl")
    testfile("tangent.jl")
    testfile("poincare.jl")
    testfile("projected.jl")
end