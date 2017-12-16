cd()
println("\nTesting Dataset IO (file at $(pwd()))...")
if current_module() != DynamicalSystemsDef
  using DynamicalSystemsDef
end
using Base.Test, StaticArrays

@testset "Dataset IO" begin
  ds = Systems.towel()
  data = trajectory(ds, 1000)

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
