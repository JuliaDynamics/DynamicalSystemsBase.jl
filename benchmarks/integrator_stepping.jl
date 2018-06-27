using DynamicalSystemsBase
using Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac
using BenchmarkTools

u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = zeros(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]

for i in 1:8
    println("Time to create & step! integrators, combination $i")
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

    integ = integrator(ds)
    println("normal")
    @time integ = integrator(ds)
    step!(integ)
    @time step!(integ)
    te = tangent_integrator(ds, orthonormal(3,3))
    println("tangent")
    @time te = tangent_integrator(ds, orthonormal(3,3))
    step!(te); step!(te)
    @time step!(te)
    te = parallel_integrator(ds, [INITCOD[sysindx], INITCOD[sysindx]])
    println("parallel")
    @time te = parallel_integrator(ds, [INITCOD[sysindx], INITCOD[sysindx]])
    step!(te); step!(te)
    @time step!(te)
end
