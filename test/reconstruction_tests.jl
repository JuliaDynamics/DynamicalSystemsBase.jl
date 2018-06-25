using DynamicalSystemsBase
using Test, StaticArrays

println("\nTesting Reconstruction")

ds = Systems.towel()
data = trajectory(ds, 10000)
s = data[:, 1]; N = length(s)

@testset "Reconstruction" begin

	@testset "D = $(D), τ = $(τ)" for D in [2,3], τ in [2,3]

		R = Reconstruction(s, D, τ)

		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
		@test size(R) == (length(s) - τ*(D - 1), D)
	end
end

@testset "Multi-time R" begin

    D = 2
    τ1 = [0, 2]
    τ2 = [2, 4]

    R0 = Reconstruction(s, D, 2)
    R1 = Reconstruction(s, D, τ1)
    R2 = Reconstruction(s, D, τ2)

    @test R1 == R0

    R2x = R2[:, 1]
    @test R2[:, 1] == R0[3:end, 1]
    @test R2[:, 2] == R0[3:end, 2]
    @test R2.delay[1] == 2
    @test size(R2) == (N-4, 2)

end

@testset "Multidim R" begin
    @testset "D = $(D), τ = $(τ), base = $(basedim)" for     D in [2,3], τ in [2,3], basedim in [2,3]

        si = Matrix(data[:,1:basedim])
        s = Size(10001,basedim)(si)
        R = Reconstruction(s, D, τ)
        tr = Dataset(si)
        R2 = Reconstruction(tr, D, τ)

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
    R1 = Reconstruction(data2, 4, taus)
    R2 = Reconstruction(data3, 4, taus)

    @test R1 == R2
    @test R1[:, 1] == data2[1:end-8, 1]
    @test R1[:, 2] == data2[1:end-8, 2]
    @test R1[:, 3] == data2[3:end-6, 1]

    # test error throws:
    taus = [0 0 0; 2 3 0; 4 6 0; 6 8 0]
    try
        R1 = Reconstruction(data2, 4, taus)
    catch err
        @test isa(err, ArgumentError)
        @test contains(err.msg, "delay matrix")
    end

end
