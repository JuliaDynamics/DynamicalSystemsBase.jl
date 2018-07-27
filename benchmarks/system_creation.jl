using DynamicalSystemsBase
using Test, StaticArrays
using DynamicalSystemsBase: CDS, DDS
using DynamicalSystemsBase.Systems: hoop, hoop_jac, hiip, hiip_jac
using DynamicalSystemsBase.Systems: loop, loop_jac, liip, liip_jac

u0 = [0, 10.0, 0]
p = [10, 28, 8/3]
u0h = zeros(2)
ph = [1.4, 0.3]

FUNCTIONS = [liip, liip_jac, loop, loop_jac, hiip, hiip_jac, hoop, hoop_jac]
INITCOD = [u0, u0h]
PARAMS = [p, ph]

#######################################################################################
#                                      TIMINGS                                        #
#######################################################################################
println("\nStarting timings (using @time)..")

println("\nAutodiff CDS (create/jacobian):")
println("IIP")
@time lip = CDS(liip, u0, p)
jacobian(lip); @time jacobian(lip)
println("OOP")
@time lop = CDS(loop, u0, p)
jacobian(lop); @time jacobian(lop)

println("\n CDS (create/jacobian):")
println("IIP")
@time lipjac = CDS(liip, u0, p, liip_jac)
jacobian(lipjac); @time jacobian(lipjac)
println("OOP")
@time lopjac = CDS(loop, u0, p, loop_jac)
jacobian(lopjac); @time jacobian(lopjac)

println("\nAutodiff DDS (create/jacobian):")
println("IIP")
@time hip = DDS(hiip, u0h, ph)
jacobian(hip); @time jacobian(hip)
println("OOP")
@time hop = DDS(hoop, u0h, ph)
jacobian(hop); @time jacobian(hop)

println("\nDDS (create/jacobian):")
println("IIP")
@time hipjac = DDS(hiip, u0h, ph, hiip_jac)
jacobian(hipjac); @time jacobian(hipjac)
println("OOP")
@time hopjac = DDS(hoop, u0h, ph, hoop_jac)
jacobian(hopjac); @time jacobian(hopjac)
