using Test
using DynamicalSystemsBase
using OrdinaryDiffEq: Vern9

@testset "successful_step tests -> unstable systems" begin

	############## define unstable dynamical rules ##################
	#exp_rule_discrete(x, p, n) = SVector(p[1]*Base.exp(x[1]), p[2]*Base.exp(-x[2]))
	exp_rule(x, p, t) = SVector(p[1]*Base.exp(x[1]), p[2]*Base.exp(-x[2]))
	u0 = ones(2)
	p0 = [1.0, 2.0]
	
	#discrete
	dim = DeterministicIteratedMap(exp_rule, u0, p0) 
	
	#continuous
	diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)
	ode = CoupledODEs(exp_rule, u0, p0; diffeq)
	
	#projected
	projection = [1]
	complete_state = [0.0]
	pr = ProjectedDynamicalSystem(deepcopy(ode), projection, complete_state)
	
	#parallel
	states = [u0,u0]
	pds = ParallelDynamicalSystem(deepcopy(ode),states)

	#tangent
	tds = TangentDynamicalSystem(deepcopy(ode))
	
	#stroboscopic
	sm = StroboscopicMap(deepcopy(ode),0.5)
	
	#stepping and testing
	for sys in [dim,ode,pr,pds,tds,sm]
		for _ in 1:100
			step!(sys)
		end
		@test successful_step(sys) == false
	end
end

