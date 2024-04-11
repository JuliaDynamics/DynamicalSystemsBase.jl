using DynamicalSystemsBase, Test
using ModelingToolkit

# Make the same MTK model as in the basic tutorial
@variables t
D = Differential(t)

function fol_factory(separate = false; name)
    @parameters τ
    @variables t x(t) f(t) RHS(t)

    eqs = separate ? [RHS ~ (f - x) / τ,
        D(x) ~ RHS] :
          D(x) ~ (f - x) / τ

    ODESystem(eqs, t; name)
end

@named fol_1 = fol_factory()
@named fol_2 = fol_factory(true) # has observable RHS

connections = [fol_1.f ~ 1.5,
    fol_2.f ~ fol_1.x]

connected = compose(ODESystem(connections, t; name = :connected), fol_1, fol_2)

sys = structural_simplify(connected; split = false)

u0 = [fol_1.x => -0.5,
    fol_2.x => 1.0]

p = [fol_1.τ => 2.0,
    fol_2.τ => 4.0]

prob = ODEProblem(sys, u0, (0.0, 10.0), p)
ds = CoupledODEs(prob)

# parameters
@test current_parameter(ds, 1) == 2.0
@test current_parameter(ds, fol_1.τ) == 2.0
@test current_parameter(ds, 2) == 4.0
@test current_parameter(ds, fol_2.τ) == 4.0

set_parameter!(ds, 1, 3.0)
@test current_parameter(ds, 1) == 3.0
@test current_parameter(ds, fol_1.τ) == 3.0

set_parameter!(ds, fol_1.τ, 2.0)
@test current_parameter(ds, 1) == 2.0
@test current_parameter(ds, fol_1.τ) == 2.0

# pure parameter container
pp = deepcopy(current_parameters(ds))
set_parameter!(ds, fol_1.τ, 4.0, pp)
@test current_parameter(ds, fol_1.τ, pp) == 4.0

# states and observed variables
@test observe_state(ds, 1) == -0.5
@test observe_state(ds, fol_1.x) == -0.5
@test observe_state(ds, fol_2.RHS) == -0.375

set_state!(ds, 1.5, 1)
@test observe_state(ds, 1) == 1.5
set_state!(ds, -0.5, fol_1.x)
@test observe_state(ds, 1) == -0.5

# test that derivative dynamical systems also work as execpted
u1 = current_state(ds)
pds = ParallelDynamicalSystem(ds, [u1, copy(u1)])

set_parameter!(pds, fol_1.τ, 4.0)
@test current_parameter(pds, 1) == 4.0
@test current_parameter(pds, fol_1.τ) == 4.0
@test observe_state(pds, fol_1.x) == -0.5
@test observe_state(pds, fol_2.RHS) == -0.375

sds = StroboscopicMap(ds, 1.0)
set_parameter!(sds, fol_1.τ, 2.0)
@test current_parameter(sds, 1) == 2.0
@test current_parameter(sds, fol_1.τ) == 2.0
@test observe_state(sds, fol_1.x) == -0.5
@test observe_state(sds, fol_2.RHS) == -0.375

prods = ProjectedDynamicalSystem(ds, [1], [0.0])
set_parameter!(prods, fol_1.τ, 3.0)
@test current_parameter(prods, 1) == 3.0
@test current_parameter(prods, fol_1.τ) == 3.0
@test observe_state(prods, fol_1.x) == -0.5
@test observe_state(prods, fol_2.RHS) == -0.375

# notice this evolves the dynamical system!!!
pmap = PoincareMap(ds, (1, 0.0))
set_parameter!(pmap, fol_1.τ, 4.0)
@test current_parameter(pmap, 1) == 4.0
@test current_parameter(pmap, fol_1.τ) == 4.0
@test observe_state(pmap, fol_1.x) ≈ 0 atol = 1e-3 rtol = 0

# test with split
sys = structural_simplify(connected; split = true)

u0 = [fol_1.x => -0.5,
    fol_2.x => 1.0]

p = [fol_1.τ => 2.0,
    fol_2.τ => 4.0]

prob = ODEProblem(sys, u0, (0.0, 10.0), p)
ds = CoupledODEs(prob)

@test current_parameter(ds, fol_1.τ) == 2.0
set_parameter!(ds, fol_1.τ, 3.0)
@test current_parameter(ds, fol_1.τ) == 3.0

# test without sys
function lorenz!(du, u, p, t)
    du[1] = p[1] * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
p0 = [10.0]
prob = ODEProblem(lorenz!, u0, tspan, p0)
ds = CoupledODEs(prob)

@test current_parameter(ds, 1) == 10.0
set_parameter!(ds, 1, 2.0)
@test current_parameter(ds, 1) == 2.0

@test observe_state(ds, 1) == 1.0

@test_throws ErrorException observe_state(ds, fol_1.f)

# Test that remake works also without anything initial

@variables t
D = Differential(t)
@mtkmodel Roessler begin
    @parameters begin
        a = 0.2
        b = 0.2
        c = 5.7
    end
    @variables begin
        x(t) = 1.0
        y(t) = 0.0
        z(t) = 0.0
        nlt(t) # nonlinear term
    end
    @equations begin
        D(x) ~ -y -z
        D(y) ~ x + a*y
        D(z) ~ b + nlt
        nlt ~ z*(x - c)
    end
end

@mtkbuild roessler_model = Roessler()

@testset "type: $(iip)" for iip in (true, false)
    if iip
        prob = ODEProblem(roessler_model)
    else
        prob = ODEProblem{false}(roessler_model, nothing, (0.0, Inf); u0_constructor = x->SVector(x...))
    end
    ds = CoupledODEs(prob)

    @test ds isa CoupledODEs

    @testset "partial set state" begin
        set_state!(ds, Dict(:y => 0.5, :x => 0.5))
        @test observe_state(ds, :x) == 0.5
        @test observe_state(ds, :y) == 0.5
    end

    @testset "partial set parameter" begin
        set_parameters!(ds, Dict(:a => 0.6))
        @test current_parameter(ds, :a) == 0.6
    end

    @testset "reinit with partial setting" begin
        reinit!(ds, Dict(:y => 0.9)) # referring initial state
        @test observe_state(ds, :x) == 1.0
        @test observe_state(ds, :y) == 0.9

        set_state!(ds, Dict(:y => 0.5, :x => 0.5))
        reinit!(ds, Dict(:y => 0.9); reference_state = current_state(ds))
        @test observe_state(ds, :x) == 0.5
        @test observe_state(ds, :y) == 0.9
    end

    @testset "parallel dynamical system" begin
        dicts = [Dict(:y => 0.6), Dict(:x => 0.6)]
        pds = ParallelDynamicalSystem(ds, dicts)
        @test observe_state(ds, :x, current_state(pds, 1)) == 0.5
        @test observe_state(ds, :y, current_state(pds, 1)) == 0.6
        @test observe_state(ds, :x, current_state(pds, 2)) == 0.6
        @test observe_state(ds, :y, current_state(pds, 2)) == 0.9
    end

end

# %% Trajectory with mixed and time dependent indexing
η2 = 1.0
η3 = 0.3
@variables η1(t)
@variables DT(t) = 1.2 DS(t) = 1.5

@parameters η1_0 = 2.0 # starting value for η1 parameter
@parameters r_η = 0.01  # the rate that η1 changes

eqs = [
Differential(t)(DT) ~ η1 - DT - abs(DT - DS)*DT,
Differential(t)(DS) ~ η2 - η3*DS - abs(DT - DS)*DS,
η1 ~ η1_0 + r_η*t, # this symbolic variable has its own equation!
]

sys = ODESystem(eqs, t; name = :stommel)
sys = structural_simplify(sys; split = false)

prob = ODEProblem(sys)
ds = CoupledODEs(prob)

X, tvec = trajectory(ds, 10.0; Δt = 0.1, save_idxs = Any[1, 2, η1])

@test all(abs.(diff(X[:, 1])) .> 1e-8)
@test all(diff(X[:, 3]) .≈ 0.001)
