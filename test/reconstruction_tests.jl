using DynamicalSystemsBase
using Test, StaticArrays

println("\nTesting reconstruct")

data = Dataset(rand(10001,3))
s = data[:, 1]; N = length(s)

@testset "reconstruct" begin

	@testset "D = $(D), τ = $(τ)" for D in [1,2], τ in [2,3]

		R = reconstruct(s, D, τ)

		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
		@test size(R) == (length(s) - τ*D, D+1)
	end
end

@testset "Multi-time reconstruct" begin

    D = 2
    τ1 = [2, 4]
    τ2 = [4, 8]

    R0 = reconstruct(s, D, 2)
    R1 = reconstruct(s, D, τ1)
    R2 = reconstruct(s, D, τ2)

    @test R1 == R0

    R2y = R2[:, 2]
    @test R2y == R0[5:end, 1]
    @test R2[:, 1] == R0[1:end-4, 1]
    @test size(R2) == (N-maximum(τ2), 3)

end

@testset "Multidim R" begin
    @testset "D = $(D), τ = $(τ), base = $(basedim)" for     D in [2,3], τ in [2,3], basedim in [2,3]

        si = Matrix(data[:,1:basedim])
        s = Size(10001,basedim)(si)
        R = reconstruct(s, D, τ)
        tr = Dataset(si)
        R2 = reconstruct(tr, D, τ)

        for dim in 1:basedim
            @test R[(1+τ):end, dim] == R[1:end-τ, dim+basedim]
            @test R2[(1+τ):end, dim] == R[1:end-τ, dim+basedim]
        end
        @test size(R) == (size(s,1) - τ*(D - 1), D*basedim)
        @test size(R2) == (size(s,1) - τ*(D - 1), D*basedim)
    end
end

@testset "Multidim Multi-time" begin

    taus = [0 0; 2 3; 4 6; 6 8]
    data2 = data[:, 1:2]
    data3 = Size(10001, 2)(Matrix(data2))
    R1 = reconstruct(data2, 4, taus)
    R2 = reconstruct(data3, 4, taus)

    @test R1 == R2
    @test R1[:, 1] == data2[1:end-8, 1]
    @test R1[:, 2] == data2[1:end-8, 2]
    @test R1[:, 3] == data2[3:end-6, 1]

    # test error throws:
    taus = [0 0 0; 2 3 0; 4 6 0; 6 8 0]
    try
        R1 = reconstruct(data2, 4, taus)
    catch err
        @test isa(err, ArgumentError)
        @test contains(err.msg, "delay matrix")
    end

end
