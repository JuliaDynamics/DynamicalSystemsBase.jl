using Test
using DynamicalSystemsBase

@testset "successful_step tests -> unstable systems" begin

	############## define unstable dynamical rules ##################
	#exp_rule_discrete(x, p, n) = SVector(p[1]*Base.exp(x[1]), p[2]*Base.exp(-x[2]))
	exp_rule(x, p, t) = SVector(p[1]*Base.exp(x[1]), p[2]*Base.exp(-x[2]))
	u0 = ones(2)
	p0 = [1.0, 2.0]

	dim = DeterministicIteratedMap(exp_rule, u0, p0)

	diffeq = (adaptive = false, dt = 10.0)
	ode = CoupledODEs(exp_rule, u0, p0; diffeq)

	projection = [1]
	complete_state = [0.0]
	pr = ProjectedDynamicalSystem(deepcopy(ode), projection, complete_state)

	pds = ParallelDynamicalSystem(deepcopy(ode), [u0,u0])

	tds = TangentDynamicalSystem(deepcopy(ode))

	sm = StroboscopicMap(deepcopy(ode), 20.0)

	@testset "$(nameof(typeof(sys)))" for sys in [dim,ode,pr,pds,tds,sm]
		suc = true
		nmax = 100
		n = 0
		while (suc == true) && (n < nmax)
			step!(sys)
			suc = successful_step(sys)
			n += 1
		end
		@test suc == false
	end
end
