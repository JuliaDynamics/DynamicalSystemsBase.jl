using Test, StaticArrays

println("\nTesting Dataset...")

@testset "Dataset" begin
  data = Dataset(rand(1001,3))
  @testset "Methods & Indexing" begin
    a = data[:, 1]
    b = data[:, 2]
    c = data[:, 3]

    @test Dataset(a, b, c) == data
    @test size(Dataset(a, b)) == (1001, 2)

    @test data[:, 2:3][:, 1] == data[:, 2]

    @test size(data[1:10,1:2]) == (10,2)
    @test data[1:10,1:2] == Dataset(a[1:10], b[1:10])
    @test data[SVector{10}(1:10), SVector(1, 2)] == data[1:10, 1:2]
  end

  @testset "minmax" begin
    mi = minima(data)
    ma = maxima(data)
    mimi, mama = minmaxima(data)
    @test mimi == mi
    @test mama == ma
    for i in 1:3
      @test mi[i] < ma[i]
    end
  end

  @testset "Conversions" begin
    m = Matrix(data)
    @test Dataset(m) == data

    m = rand(1000, 4)
    @test Matrix(Dataset(m)) == m
  end

end
