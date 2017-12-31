cd()
println("\nTesting Dataset IO (file at $(pwd()))...")
if current_module() != DynamicalSystemsBase
  using DynamicalSystemsBase
end
using Base.Test, StaticArrays

@testset "Dataset" begin
  ds = Systems.towel()
  data = trajectory(ds, 1000)
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
  end
end
