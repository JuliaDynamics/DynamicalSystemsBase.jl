# Define a function that tests all fundamentals of a `DynamicalSystem` implementation
# All concrete subtypes must pass these tests, with the exception of whether
# `trajectory` is tested or not.
using DynamicalSystemsBase, Test

function test_dynamical_system(ds, u0, p0; idt, iip,
    test_init_state_equiv = true, test_trajectory = true, u0init = deepcopy(u0))

    @testset "obtaining info" begin
        @test current_state(ds) == u0
        @test initial_state(ds) == u0init
        @test current_parameters(ds) == p0
        @test initial_parameters(ds) == p0
        @test current_time(ds) == 0
        @test initial_time(ds) == 0
        @test isinplace(ds) == iip
        @test isdeterministic(ds) == true
        @test isdiscretetime(ds) == idt
        @test ds(0) == u0
    end

    @testset "alteration" begin
        set_state!(ds, u0 .+ 1)
        # this test doesnt work on poincare map or projected system
        # where by definition state must always be on the plane.
        if test_init_state_equiv
            @test current_state(ds) == u0 .+ 1
        end

        fpv = current_parameters(ds)[1]
        set_parameter!(ds, 1, 1.0)
        @test current_parameters(ds)[1] == 1
        set_parameters!(ds, [2.0])
        @test current_parameters(ds)[1] == 2.0
        set_parameters!(ds, [2.0, 0.1])
        @test current_parameters(ds)[2] == 0.1

        reinit!(ds; p = initial_parameters(ds))
        @test ds(0) == u0
        @test current_state(ds) == u0
        @test current_parameters(ds)[1] == fpv
    end


    @testset "time evolution" begin
        if idt
            @test_throws ArgumentError ds(2)

            # For discrete systems this is always the second state no matter what
            if ds isa DeterministicIteratedMap
                second_state = if iip
                    z = deepcopy(current_state(ds))
                    dynamic_rule(ds)(z, current_state(ds), current_parameters(ds), current_time(ds))
                    z
                else
                    dynamic_rule(ds)(current_state(ds), current_parameters(ds), current_time(ds))
                end
                step!(ds) # notice that `ds` has been `reinit!` in the previous block
            else
                # Unfortunately here we don't have a way to get "guaranteed"
                # what the second state would be in the analytic sense... so we cheat
                step!(ds)
                second_state = deepcopy(current_state(ds))
            end

            @test current_time(ds) == 1
            @test current_state(ds) == second_state
            @test ds(1) == second_state
            step!(ds, 2)
            @test current_time(ds) == 3
            @test current_state(ds) != second_state != u0
            step!(ds, 1, true)
            @test current_time(ds) == 4
        else
            t0 = current_time(ds)
            xi = deepcopy(current_state(ds)[1])
            step!(ds)
            t1 = current_time(ds)
            @test t1 > t0

            tm = (t1 - t0)/2
            xm = ds(tm)[1]
            xf = current_state(ds)[1]
            @test min(xi, xf) ≤ xm ≤ max(xi, xf)

            step!(ds, 1.0)
            @test current_time(ds) ≥ t1 + 1

            t2 = current_time(ds)
            step!(ds, 1.0, true)
            @test current_time(ds) == t2 + 1.0
        end

        if test_trajectory

        @testset "trajectory" begin
            if idt
                reinit!(ds)
                @test current_state(ds) == u0
                X, t = trajectory(ds, 10)
                @test X isa Dataset{dimension(ds), Float64}
                @test X[1] == u0
                @test X[2] == second_state
                @test t == 0:1:10
                @test length(X) == length(t) == 11

                # Continue as is from current state:
                Y, t = trajectory(ds, 10, nothing)
                @test t[1] == 10
                @test Y[1] == X[end]

                # obtain only first variable
                Z, t = trajectory(ds, 10; save_idxs = [1])
                @test size(Z) == (11, 1)
                @test Z[1][1] == u0[1]
            else
                reinit!(ds)
                @test current_state(ds) == u0
                X, t = trajectory(ds, 3; Δt = 0.1)
                @test Base.step(t) == 0.1
                @test t[1] == initial_time(ds)
                @test X isa Dataset{dimension(ds), Float64}
                @test X[1] == u0

                prev_u0 = deepcopy(current_state(ds))
                Y, t2 = trajectory(ds, 3, nothing; Δt = 1)
                @test Y[1] ≈ prev_u0 atol=1e-6
                @test t2[1] > t[end]

                # obtain only first variable
                Z, t = trajectory(ds, 3; save_idxs = [1], Δt = 1)
                @test size(Z) == (4, 1)
                @test Z[1][1] == u0[1]
            end
        end
        end

    end

end

