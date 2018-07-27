using Test, StaticArrays

println("\nTesting reconstruct")

@testset "reconstruct" begin

    data = Dataset(rand(10001,3))
    s = data[:, 1]; N = length(s)

    @testset "standard" begin

    	@testset "D = $(D), τ = $(τ)" for D in [1,2], τ in [2,3]

    		R = reconstruct(s, D, τ)

    		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
    		@test size(R) == (length(s) - τ*D, D+1)
    	end
    end

    @testset "multi-time" begin

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

        @test_throws ArgumentError reconstruct(s, 4, τ1)


    end

    @testset "multidim " begin
        @testset "D = $(D), B = $(B)" for  D in [2,3], B in [2,3]

            τ = 3
            si = Matrix(data[:,1:B])
            sizedsi = Size(N,B)(si)
            R = reconstruct(sizedsi, D, τ)
            tr = Dataset(si)
            R2 = reconstruct(tr, D, τ)

            @test R == R2

            for dim in 1:B
                @test R[(1+τ):end, dim] == R[1:end-τ, dim+B]
                @test R2[(1+τ):end, dim] == R[1:end-τ, dim+B]
            end
            @test size(R) == (size(s,1) - τ*D, (D+1)*B)
        end
    end

    @testset "multidim multi-time" begin

        taus = [2 3; 4 6; 6 8]
        data2 = data[:, 1:2]
        data3 = Size(N, 2)(Matrix(data2))
        R1 = reconstruct(data2, 3, taus)
        R2 = reconstruct(data3, 3, taus)

        @test R1 == R2
        @test R1[:, 1] == data2[1:end-8, 1]
        @test R1[:, 2] == data2[1:end-8, 2]
        @test R1[:, 3] == data2[3:end-6, 1]

        # test error throws:
        taus = [0 0 0; 2 3 0; 4 6 0; 6 8 0]
        @test_throws ArgumentError reconstruct(data2, 5, taus)
        @test_throws ArgumentError reconstruct(data2, 4, taus)

    end
end
