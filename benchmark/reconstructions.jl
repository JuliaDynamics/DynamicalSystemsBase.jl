SUITE["Reconstructions"] = BenchmarkGroup()
SR = SUITE["Reconstructions"]

srand(1234)
N = 10000; D = 3; B = 3

data = Dataset(rand(N, B))
x = data[:, 1]

combinations = ["standard", "multitime", "multidim", "allmulti"]
taus = (3, [2,7,8], 3, hcat([2,7,8],[2, 5,9],[4,8,12]))

for i in 1:4
    SR[combinations[i]] = BenchmarkGroup()
    if i < 3
        SR[combinations[i]]["full"] = @benchmarkable reconstruct($x, $D, $(taus[i]))
        de = DelayEmbedding(D, taus[i])
        SR[combinations[i]]["vector"] = @benchmarkable $(de)($x, 1)
    else
        SR[combinations[i]]["full"] = @benchmarkable reconstruct($data, $D, $(taus[i]))
        de = MTDelayEmbedding(D, taus[i], B)
        SR[combinations[i]]["vector"] = @benchmarkable $(de)($data, 1)
    end
end
