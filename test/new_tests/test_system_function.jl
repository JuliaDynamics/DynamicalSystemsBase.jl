using DynamicalSystemsBase, Test

function test_dynamical_system(ds, u0, name, idt, iip)

    @testset "$(name), idt=$(idt), iip=$(iip)" begin

        @testset "obtaining info" begin
            @test current_state(ds) == u0
            @test initial_state(ds) == u0
            @test current_parameters(ds) == p0
            @test initial_parameters(ds) == p0
            @test current_time(ds) == 0
            @test initial_time(ds) == 0
            @test isinplace(ds) == iip
            @test isdeterministic(ds) == true
            @test isdiscretetime(ds) == idt
            @test ds(0) == u0
        end

        @testset "time evolution" begin
            if idt
                @test_throws ArgumentError ds(2)
                step!(ds)
                @test current_time(ds) == 1
                @test current_state(ds) == [1, 0]
                @test ds(1) == [1, 0]
                step!(ds, 2)
                @test current_time(ds) == 3
                @test current_state(ds) != [1, 0] != u0
                step!(ds, 1, true)
                @test current_time(ds) == 4
            else
                # lalala
            end
        end

        @testset "alteration" begin
            set_state!(ds, u0 .+ 1)
            @test current_state(ds) == u0 .+ 1

            fpv = current_parameters(ds)[1]
            set_parameter!(ds, 1, 1.0)
            @test current_parameters(ds)[1] == 1
            set_parameters!(ds, [2.0])
            @test current_parameters(ds)[1] == 2.0
            set_parameters!(ds, [2.0, 0.1])
            @test current_parameters(ds)[2] == 0.1

            reinit!(ds; p0 = initial_parameters(ds))
            @test ds(0) == u0
            @test current_state(ds) == u0
            @test current_parameters(ds)[1] == fpv
        end

        @testset "trajectory" begin
            if idt
                reinit!(ds)
                @test current_state(ds) == u0
                X, t = trajectory(ds, 100)
                @test X isa Dataset{2, Float64}
                @test X[1] == u0
                @test t == 0:1:100
                @test length(X) == length(t) == 101

                # Continue as is from current state:
                Y, t = trajectory(ds, 100, nothing)
                @test t[1] == 100
                @test Y[1] == X[end]

                second_state = (reinit!(ds); step!(ds); current_state(ds))
                @test X[2] == second_state

                # obtain only first variable
                Z, t = trajectory(ds, 100; save_idxs = [1])
                @test size(Z) == (101, 1)
                @test Z[1][1] == u0[1]
            else
                # lalala
            end
        end

    end
end

