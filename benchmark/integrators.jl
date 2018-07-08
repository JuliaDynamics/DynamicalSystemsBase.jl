SUITE["Integrators"] = BenchmarkGroup()

SI = SUITE["Integrators"]

using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: eom_towel, jacob_towel, eom_towel_iip, jacob_towel_iip
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac

u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0t = ones(2)
pt = nothing

FUNCTIONS = (liip, liip_jac, loop, loop_jac, eom_towel_iip, jacob_towel_iip, eom_towel, jacob_towel)
INITCOD = (u0, u0t)
PARAMS = (p, pt)

combinations = ["C_I", "C_I_J", "C_O", "C_O_J", "D_I", "D_I_J", "D_O", "D_O_J"]

for i in 1:8
    SI[combinations[i]] = BenchmarkGroup()
    sysindx = i < 5 ? 1 : 2
    if i < 5
        if isodd(i)
            ds = CDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
        else
            ds = CDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
        end
    else
        if isodd(i)
            ds = DDS(FUNCTIONS[i], INITCOD[sysindx], PARAMS[sysindx])
        else
            ds = DDS(FUNCTIONS[i-1], INITCOD[sysindx], PARAMS[sysindx], FUNCTIONS[i])
        end
    end

    tinteg = tangent_integrator(ds); step!(tinteg)

    if isodd(i)
        integ = integrator(ds)
        pinteg = parallel_integrator(ds, [INITCOD[sysindx], INITCOD[sysindx]])
        step!(pinteg); step!(integ)
        SI[combinations[i]]["integ"] = @benchmarkable step!($integ)
        SI[combinations[i]]["pinteg"] = @benchmarkable step!($pinteg)
    end
    SI[combinations[i]]["tinteg"] = @benchmarkable step!($tinteg)
end
