using DynamicalSystemsBase, SimpleDiffEq, DiffEqBase
using Statistics, Test

ds = Systems.lorenz()
ALG = SimpleATsit5()

# %%

i1 = tangent_integrator(ds, 3; alg = ALG)
i2 = tangent_integrator(ds, 3; alg = ALG,
internalnorm = DiffEqBase.ODE_DEFAULT_NORM)

step!(i1)
step!(i2)
dts = zeros(2, 500)
for i in 1:500
    step!(i1); step!(i2)
    dts[:, i] .= (i1.dt, i2.dt)
end

dt1 = mean(dts[1, :])
dt2 = mean(dts[2, :])

@test dt1 > dt2

# %%
s = [get_state(ds), get_state(ds) .+ rand(3)]
i1 = parallel_integrator(ds, deepcopy(s);
alg = ALG, internalnorm = DiffEqBase.ODE_DEFAULT_NORM)
i2 = parallel_integrator(ds, deepcopy(s);
alg = ALG)

step!(i1)
step!(i2)
dts = zeros(2, 1000)
for i in 1:1000
    step!(i1); step!(i2)
    dts[:, i] .= (i1.dt, i2.dt)
end

dt1 = mean(dts[1, :])
dt2 = mean(dts[2, :])

@test dt2 > dt1
