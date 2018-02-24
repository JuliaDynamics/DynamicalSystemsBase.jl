println("\nTesting Reconstruction")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

@testset "Reconstruction" begin
	ds = Systems.towel()
	data = trajectory(ds, 10000)
	s = data[:, 1]; N = length(s)
	@testset "D = $(D), τ = $(τ)" for D in [2,3], τ in [2,3]

		R = Reconstruction(s, D, τ)

		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
		@test size(R) == (length(s) - τ*(D - 1), D)
	end

    @testset "D = $(D), τ = $(τ), basedim = $(basedim)" for D in [2,3], τ in [2,3], basedim in [2,3]

        si = Matrix(data[:,1:basedim])
        s = Size(10001,basedim)(si)
        R = Reconstruction(s, D, τ)

        for dim in 1:basedim
            @test R[(1+τ):end, dim] == R[1:end-τ, dim+basedim]
        end
        @test size(R) == (size(s,1) - τ*(D - 1), D*basedim)
    end
end
