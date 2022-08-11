using Test
using DynamicalSystemsBase
using DynamicalSystemsBase:orthonormal
using ChaosTools
using OrdinaryDiffEq

#test succesful_step for continuous and discrete ds
#stable and unstable cases

@testset "successful_step tests -> unstable systems" begin

	############## define unstable dynamical systems ##################
	function divergent_chaotictest!(du,u, p, t)
           k1, k2 = p
           x, y, z = u
           du[1] = y
           du[2] = z
           du[3] = k1-7*y+x^2+k2*z
	end

	u0 = rand(3)
	para = [2, -0.9] #k2=k6=0

	ds_cont_unstable = ContinuousDynamicalSystem(divergent_chaotictest!, u0, para)
	ds_disc_unstable = DiscreteDynamicalSystem(divergent_chaotictest!, u0, para)
	
	
	T = 50
	diffeq = (alg=Tsit5(),)
	
	######################### make integrators #######################
	
	cont_unstable_integ = integrator(ds_cont_unstable,get_state(ds_cont_unstable);diffeq)
	disc_unstable_integ = integrator(ds_disc_unstable,get_state(ds_disc_unstable))
	
	cont_unstable_pinteg = parallel_integrator(ds_cont_unstable,[get_state(ds_cont_unstable),get_state(ds_cont_unstable)];diffeq)
	disc_unstable_pinteg = parallel_integrator(ds_disc_unstable,[get_state(ds_disc_unstable),get_state(ds_disc_unstable)])	
	
	cont_unstable_tinteg = tangent_integrator(ds_cont_unstable,orthonormal(dimension(ds_cont_unstable),dimension(ds_cont_unstable));diffeq)
	disc_unstable_tinteg = tangent_integrator(ds_disc_unstable,orthonormal(dimension(ds_cont_unstable),dimension(ds_cont_unstable)))

	@test_logs (:warn, "Instability detected. Aborting")  step!(cont_unstable_integ, T)
	@test successful_step(cont_unstable_integ) == false
	
	step!(disc_unstable_integ, T)
	@test successful_step(disc_unstable_integ) == false 
	
	@test_logs (:warn, "Instability detected. Aborting")  step!(cont_unstable_pinteg, T)
	@test successful_step(cont_unstable_pinteg) == false
	
	step!(disc_unstable_pinteg, T)
	@test successful_step(disc_unstable_pinteg) == false 
	
	@test_logs (:warn, "Instability detected. Aborting")  step!(cont_unstable_tinteg, T)
	@test successful_step(cont_unstable_tinteg) == false
	
	step!(disc_unstable_tinteg, T)
	@test successful_step(disc_unstable_tinteg) == false 
	
end


@testset "successful_step tests -> stable systems" begin

	T = 50
	diffeq = (alg=Tsit5(),)
	

	################### define stable dynamical systems ####################
	
	ds_cont_stable = Systems.lorenz()
	ds_disc_stable = Systems.henon()


	#################### make integrators ######################
	
	cont_stable_integ = integrator(ds_cont_stable,get_state(ds_cont_stable);diffeq)
	disc_stable_integ = integrator(ds_disc_stable,get_state(ds_disc_stable))

	cont_stable_pinteg = parallel_integrator(ds_cont_stable,[get_state(ds_cont_stable),get_state(ds_cont_stable)];diffeq)
	disc_stable_pinteg = parallel_integrator(ds_disc_stable,[get_state(ds_disc_stable),get_state(ds_disc_stable)])	
	
	cont_stable_tinteg = tangent_integrator(ds_cont_stable,orthonormal(dimension(ds_cont_stable),dimension(ds_cont_stable));diffeq)
	disc_stable_tinteg = tangent_integrator(ds_disc_stable,orthonormal(dimension(ds_disc_stable),dimension(ds_disc_stable)))

	step!(cont_stable_integ, T)
	@test successful_step(cont_stable_integ)
	
	step!(disc_stable_integ, T)
	@test successful_step(disc_stable_integ)
	

	step!(cont_stable_integ, T)
	@test successful_step(cont_stable_integ)
	
	step!(disc_stable_integ, T)
	@test successful_step(disc_stable_integ) 
	
	step!(cont_stable_pinteg, T)
	@test successful_step(cont_stable_pinteg)
	
	step!(disc_stable_pinteg, T)
	@test successful_step(disc_stable_pinteg) 
	
	step!(cont_stable_tinteg, T)
	@test successful_step(cont_stable_tinteg) 
	
	step!(disc_stable_tinteg, T)
	@test successful_step(disc_stable_tinteg)  
	
end


