using DynamicalSystemsBase
using Statistics, Test

ds = Systems.lorenz()
ALG = SimpleATsit5()

println("\nTesting internal norms...")


# %%
@testset begin "internal norm: tangent"
i1 = tangent_integrator(ds, 3; diffeq = (alg = ALG,))
i2 = tangent_integrator(ds, 3; diffeq = (alg = ALG, internalnorm = DynamicalSystemsBase._standardnorm))

step!(i1)
step!(i2)
N = 1000
dts = zeros(2, N)
for i in 1:N
    step!(i1); step!(i2)
    dts[:, i] .= (i1.dt, i2.dt)
end

dt1 = mean(dts[1, :])
dt2 = mean(dts[2, :])

@test dt1 > dt2
end
# %%
@testset begin "internal norm: parallel"
s = [get_state(ds), get_state(ds) .+ rand(3)]
i1 = parallel_integrator(ds, deepcopy(s);
diffeq = (alg = ALG, internalnorm = DynamicalSystemsBase._standardnorm))
i2 = parallel_integrator(ds, deepcopy(s); diffeq = (alg = ALG,))

step!(i1)
step!(i2)

N = 2000
dts = zeros(2, N)
for i in 1:N
    step!(i1); step!(i2)
    dts[:, i] .= (i1.dt, i2.dt)
end

dt1 = mean(dts[1, :])
dt2 = mean(dts[2, :])

@test dt2 > dt1
end