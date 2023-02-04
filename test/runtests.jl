using DynamicalSystemsBase
using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "DynamicalSystemsBase tests" begin
    testfile("new_tests/discrete.jl")
    testfile("new_tests/continuous.jl")
    testfile("new_tests/stroboscopic.jl")
    testfile("new_tests/parallel.jl")
    testfile("new_tests/tangent.jl")
end
