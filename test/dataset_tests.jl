cd()
println("\nTesting Dataset IO (file at $(pwd()))...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

@testset "Dataset" begin
  ds = Systems.towel()
  data = trajectory(ds, 1000)

  @testset "Methods & Indexing" begin
    a = data[:, 1]
    b = data[:, 2]
    c = data[:, 3]

    @test Dataset(a, b, c) == data
    @test size(Dataset(a, b)) == (1000, 2)

    @test data[:, 2:3][:, 1] == data[:, 2]
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
    @test reinterpret(Dataset, reinterpret(Matrix, data)) == data
    @test transpose(m) == reinterpret(Matrix, data)

    m = rand(1000, 4)
    @test reinterpret(Matrix, reinterpret(Dataset, m)) == m
    @test Dataset(m) !== Dataset(transpose(m))
    @test Dataset(m) == reinterpret(Dataset, transpose(m))
  end

  @testset "IO" begin
    @test !isfile("test.txt")
    write_dataset("test.txt", data)
    @test isfile("test.txt")

    data3 = read_dataset("test.txt", Dataset{3, Float64})
    @test dimension(data3) == 3
    @test data3 == data

    data2 = read_dataset("test.txt", Dataset{2, Float64})
    @test dimension(data2) == 2

    rm("test.txt")

    write_dataset("test.txt", data, ',')
    @test isfile("test.txt")

    data3 = read_dataset("test.txt", Dataset{3, Float64}, ',')
    @test dimension(data3) == 3
    @test data3 == data

    data2 = read_dataset("test.txt", Dataset{2, Float64})
    @test dimension(data2) == 2

    rm("test.txt")

    @test !isfile("test.txt") # make extra sure!
  end
end
