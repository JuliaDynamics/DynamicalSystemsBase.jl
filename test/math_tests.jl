if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Test, StaticArrays
println("Testing custom QR-decomposition...")

@testset "QR-decomposition" begin
    tol = 1e-10
    @testset "Known matrix" begin
        A = [1. 1 0;
             1  0 1;
             0  1 1]

        QA, RA = DynamicalSystemsBase.qr_sq(A)
        QtA =  [1/√2   1/√6  -1/√3;
                1/√2  -1/√6   1/√3;
                0      2/√6   1/√3]
        RtA =  [2/√2,  3/√6,  2/√3]
        for i in length(QA)
            @test isapprox(abs(QA[i]), abs(QtA[i]), rtol = tol)
        end
        for i in 1:3
            @test abs(RA[i,i]) ≈ abs(RtA[i])
        end
    end
    @testset "Random matrix" begin
        for i in 1:10

            A = rand(5,5)
            QA, RA = DynamicalSystemsBase.qr_sq(A)
            QtA, RtA = qr(A)
            for i in length(QA)
                @test isapprox(abs(QA[i]), abs(QtA[i]), rtol = tol)
            end
            for i in 1:3
                @test isapprox(abs(RA[i,i]), abs(RtA[i,i]), rtol = tol)
            end
        end
    end
end
