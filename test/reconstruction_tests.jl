using DynamicalSystemsBase
using Base.Test, StaticArrays

println("\nTesting Reconstruction")

ds = Systems.towel()
data = trajectory(ds, 10000)
s = data[:, 1]; N = length(s)

@testset "Reconstruction" begin

	@testset "D = $(D), τ = $(τ)" for D in [2,3], τ in [2,3]

		R = Reconstruction(s, D, τ)

		@test R[(1+τ):end, 1] == R[1:end-τ, 2]
		@test size(R) == (length(s) - τ*(D - 1), D)
	end
end

@testset "Multi-time R" begin

    D = 2
    τ1 = [0, 2]
    τ2 = [2, 4]

    R0 = Reconstruction(s, D, 2)
    R1 = Reconstruction(s, D, τ1)
    R2 = Reconstruction(s, D, τ2)

    @test R1 == R0

    R2x = R2[:, 1]
    @test R2[:, 1] == R0[3:end, 1]
    @test R2[:, 2] == R0[3:end, 2]
    @test R2.delay[1] == 2
    @test size(R2) == (N-4, 2)

end

@testset "Multidim R" begin
    @testset "D = $(D), τ = $(τ), base = $(basedim)" for     D in [2,3], τ in [2,3], basedim in [2,3]

        si = Matrix(data[:,1:basedim])
        s = Size(10001,basedim)(si)
        R = Reconstruction(s, D, τ)
        tr = Dataset(si)
        R2 = Reconstruction(tr, D, τ)

        for dim in 1:basedim
            @test R[(1+τ):end, dim] == R[1:end-τ, dim+basedim]
            @test R2[(1+τ):end, dim] == R[1:end-τ, dim+basedim]
        end
        @test size(R) == (size(s,1) - τ*(D - 1), D*basedim)
        @test size(R2) == (size(s,1) - τ*(D - 1), D*basedim)
    end
end

@testset "Multidim Multi-time" begin

    taus = [0 0; 2 3; 4 6; 6 8]
    data2 = data[:, 1:2]
    data3 = Size(10001, 2)(Matrix(data2))
    R1 = Reconstruction(data2, 4, taus)
    R2 = Reconstruction(data3, 4, taus)

    @test R1 == R2
    @test R1[:, 1] == data2[1:end-8, 1]
    @test R1[:, 2] == data2[1:end-8, 2]
    @test R1[:, 3] == data2[3:end-6, 1]

    # test error throws:
    taus = [0 0 0; 2 3 0; 4 6 0; 6 8 0]
    try
        R1 = Reconstruction(data2, 4, taus)
    catch err
        @test isa(err, ArgumentError)
        @test contains(err.msg, "delay matrix")
    end

end


@testset "Estimate Delay" begin

    ds = Systems.henon()
    data = trajectory(ds,100)
    x = data[:,1]
    @test estimate_delay(x,"first_zero") <= 2
    @test estimate_delay(x,"first_min")  <= 2
    @test estimate_delay(x,"exp_decay")  <= 2

    ds = Systems.roessler()
    dt = 0.01
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4

    dt = 0.1
    data = trajectory(ds,2000,dt=dt)
    x = data[:,1]
    @test 1.3 <= estimate_delay(x,"first_zero")*dt <= 1.7
    @test 2.6 <= estimate_delay(x,"first_min")*dt  <= 3.4


    # ds = Systems.lorenz()
    #
    # dt = 0.01
    # data = trajectory(ds,2000;dt=dt)
    # x = data[500:end,1]
    # println(estimate_delay(x,"exp_decay"))
    # #plot(autocor(x, 0:length(x)÷10, demean=true))
    # @test 2.5 <= estimate_delay(x,"exp_decay")*dt  <= 3.5
    #
    # dt = 0.1
    # data = trajectory(ds,2000;dt=dt)
    # x = data[:,1]
    # @test 2.5 <= estimate_delay(x,"exp_decay")*dt  <= 3.5
    # println(estimate_delay(x,"exp_decay"))

end

function saturation_point(Ds, E1s; threshold = 0.01, kwargs...)
    lrs, slops = linear_regions(Ds, E1s; kwargs...)
    i = findfirst(x -> x < threshold, slops)
    return i == 0 ? Ds[end] : Ds[lrs[i]]
end
function linear_regions(x::AbstractVector, y::AbstractVector;
    dxi::Int = 1, tol::Real = 0.2)

    maxit = length(x) ÷ dxi

    tangents = Float64[linreg(view(x, 1:max(dxi, 2)), view(y, 1:max(dxi, 2)))[2]]

    prevtang = tangents[1]
    lrs = Int[1] #start of first linear region is always 1
    lastk = 1

    # Start loop over all partitions of `x` into `dxi` intervals:
    for k in 1:maxit-1
        tang = linreg(view(x, k*dxi:(k+1)*dxi), view(y, k*dxi:(k+1)*dxi))[2]
        if isapprox(tang, prevtang, rtol=tol)
            # Tanget is similar with initial previous one (based on tolerance)
            continue
        else
            # Tangent is not similar.
            # Push new tangent for a new linear region
            push!(tangents, tang)

            # Set the START of a new linear region
            # which is also the END of the previous linear region
            push!(lrs, k*dxi)
            lastk = k
        end

        # Set new previous tangent (only if it was not the same as current)
        prevtang = tang
    end
    push!(lrs, length(x))
    return lrs, tangents
end


@testset "Estimate Dimension" begin
    s = sin.(0:0.1:1000)
    τ = 15
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.01) == 2


    ds = Systems.roessler();τ=15; dt=0.1
    data = trajectory(ds,1000;dt=dt)
    s = data[:,1]
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.1) == 3

    ds = Systems.lorenz();τ=5; dt=0.01
    data = trajectory(ds,500;dt=dt)
    s = data[:,1]
    D = 1:7
    E1s = estimate_dimension(s,τ,D)
    @test saturation_point(D,E1s; threshold=0.1) == 3
end
