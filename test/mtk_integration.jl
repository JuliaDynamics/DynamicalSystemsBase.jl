using DynamicalSystemsBase, Test
using ModelingToolkit
using OrdinaryDiffEq

# Make the same MTK model as in the basic tutorial
@variables t
D = Differential(t)

function fol_factory(separate = false; name)
    @parameters τ
    @variables t x(t) f(t) RHS(t)

    eqs = separate ? [RHS ~ (f - x) / τ,
        D(x) ~ RHS] :
          D(x) ~ (f - x) / τ

    ODESystem(eqs; name)
end

@named fol_1 = fol_factory()
@named fol_2 = fol_factory(true) # has observable RHS

connections = [fol_1.f ~ 1.5,
    fol_2.f ~ fol_1.x]

connected = compose(ODESystem(connections, name = :connected), fol_1, fol_2)

connected_simp = structural_simplify(connected)

u0 = [fol_1.x => -0.5,
    fol_2.x => 1.0]

p = [fol_1.τ => 2.0,
    fol_2.τ => 4.0]

prob = ODEProblem(connected_simp, u0, (0.0, 10.0), p)
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

# states and observed variables
@test observe_state(ds, 1) == -0.5
@test observe_state(ds, fol_1.x) == -0.5
@test_throws ArgumentError observe_state(ds, "test")

# %% Test that derivative dynamical systems also work as execpted
u1 = current_state(ds)
pds = ParallelDynamicalSystem(ds, [u1, copy(u1)])

set_parameter!(pds, fol_1.τ, 4.0)
@test current_parameter(pds, 1) == 4.0
@test current_parameter(pds, fol_1.τ) == 4.0
@test observe_state(pds, fol_1.x) == -0.5

sds = StroboscopicMap(ds, 1.0)
set_parameter!(sds, fol_1.τ, 2.0)
@test current_parameter(sds, 1) == 2.0
@test current_parameter(sds, fol_1.τ) == 2.0
@test observe_state(sds, fol_1.x) == -0.5

prods = ProjectedDynamicalSystem(ds, [1], [0.0])
set_parameter!(prods, fol_1.τ, 3.0)
@test current_parameter(prods, 1) == 3.0
@test current_parameter(prods, fol_1.τ) == 3.0
@test observe_state(prods, fol_1.x) == -0.5

# notice this evolves the dynamical system
pmap = PoincareMap(ds, (1, 0.0))
set_parameter!(pmap, fol_1.τ, 4.0)
@test current_parameter(pmap, 1) == 4.0
@test current_parameter(pmap, fol_1.τ) == 4.0
@test observe_state(pmap, fol_1.x) ≈ 0 atol = 1e-3 rtol = 0

# %% Test without sys
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
