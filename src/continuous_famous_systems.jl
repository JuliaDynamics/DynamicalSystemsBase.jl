"""
```julia
lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3) -> ds
```
```math
\\begin{aligned}
\\dot{X} &= \\sigma(Y-X) \\\\
\\dot{Y} &= -XZ + \\rho X -Y \\\\
\\dot{Z} &= XY - \\beta Z
\\end{aligned}
```
The famous three dimensional system due to Lorenz [1], shown to exhibit
so-called "deterministic nonperiodic flow". It was originally invented to study a
simplified form of atmospheric convection.

Currently, it is most famous for its strange attractor (occuring at the default
parameters), which resembles a butterfly. For the same reason it is
also associated with the term "butterfly effect" (a term which Lorenz himself disliked)
even though the effect applies generally to dynamical systems.
Default values are the ones used in the original paper.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : E. N. Lorenz, J. atmos. Sci. **20**, pp 130 (1963)
"""
function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CDS(loop, u0, [σ, ρ, β], loop_jac)
end
function loop(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function loop_jac(u, p, t)
    @inbounds begin
        σ, ρ, β = p
        J = @SMatrix [-σ  σ  0;
        ρ - u[3]  (-1)  (-u[1]);
        u[2]   u[1]  -β]
        return J
    end
end

function lorenz_iip(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CDS(liip, u0, [σ, ρ, β], liip_jac)
end
function liip(du, u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du[1] = σ*(u[2]-u[1])
        du[2] = u[1]*(ρ-u[3]) - u[2]
        du[3] = u[1]*u[2] - β*u[3]
        return nothing
    end
end
function liip_jac(J, u, p, t)
    @inbounds begin
    σ, ρ, β = p
    J[1,1] = -σ; J[1, 2] = σ; J[1,3] = 0
    J[2,1] = ρ - u[3]; J[2,2] = -1; J[2,3] = -u[1]
    J[3,1] = u[2]; J[3,2] = u[1]; J[3,3] = -β
    return nothing
    end
end


"""
```julia
roessler(u0=rand(3); a = 0.2, b = 0.2, c = 5.7)
```
```math
\\begin{aligned}
\\dot{x} &= -y-z \\\\
\\dot{y} &= x+ay \\\\
\\dot{z} &= b + z(x-c)
\\end{aligned}
```
This three-dimensional continuous system is due to Rössler [1].
It is a system that by design behaves similarly
to the `lorenz` system and displays a (fractal)
strange attractor. However, it is easier to analyze qualitatively, as for example
the attractor is composed of a single manifold.
Default values are the same as the original paper.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : O. E. Rössler, Phys. Lett. **57A**, pp 397 (1976)
"""
function roessler(u0=rand(3); a = 0.2, b = 0.2, c = 5.7)
    return CDS(roessler_eom, u0, [a, b, c], roessler_jacob)
end
function roessler_eom(u, p, t)
    @inbounds begin
    a, b, c = p
    du1 = -u[2]-u[3]
    du2 = u[1] + a*u[2]
    du3 = b + u[3]*(u[1] - c)
    return SVector{3}(du1, du2, du3)
    end
end
function roessler_jacob(u, p, t)
    a, b, c = p
    return @SMatrix [0.0 (-1.0) (-1.0);
                     1.0 a 0.0;
                     u[3] 0.0 (u[1]-c)]
end

"""
    double_pendulum(u0 = [π/2, 0, 0, rand()];
                    G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
Famous chaotic double pendulum system (also used for our logo!). Keywords
are gravity (G), lengths of each rod and mass of each ball (all assumed SI units).

The variables order is [θ1, dθ1/dt, θ2, dθ2/dt].

Jacobian is created automatically (thus methods that use the Jacobian will be slower)!

(please contribute the Jacobian and the e.o.m. in LaTeX :smile:)

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function double_pendulum(u0=[π/2, 0, 0, rand()]; G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
    return CDS(doublependulum_eom, u0, [G, L1, L2, M1, M2])
end
@inbounds function doublependulum_eom(state, p, t)
    G, L1, L2, M1, M2 = p

    du1 = state[2]
    del_ = state[3] - state[1]
    den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
    du2 = (M2*L1*state[2]*state[2]*sin(del_)*cos(del_) +
               M2*G*sin(state[3])*cos(del_) +
               M2*L2*state[4]*state[4]*sin(del_) -
               (M1 + M2)*G*sin(state[1]))/den1

    du3 = state[4]

    den2 = (L2/L1)*den1
    du4 = (-M2*L2*state[4]*state[4]*sin(del_)*cos(del_) +
               (M1 + M2)*G*sin(state[1])*cos(del_) -
               (M1 + M2)*L1*state[2]*state[2]*sin(del_) -
               (M1 + M2)*G*sin(state[3]))/den2
    return SVector{4}(du1, du2, du3, du4)
end

"""
    henonheiles(u0=[0, -0.25, 0.42081,0])
```math
\\begin{aligned}
\\dot{x} &= p_x \\\\
\\dot{y} &= p_y \\\\
\\dot{p}_x &= -x -2 xy \\\\
\\dot{p}_y &= -y - (x^2 - y^2)
\\end{aligned}
```

The Hénon–Heiles system [1] was introduced as a simplification of the motion
of a star around a galactic center. It was originally intended to study the
existence of a "third integral of motion" (which would make this 4D system integrable).
In that search, the authors encountered chaos, as the third integral existed
for only but a few initial conditions.

The default initial condition is a typical chaotic orbit.
The function `Systems.henonheiles_ics(E, n)` generates a grid of
`n×n` initial conditions, all having the same energy `E`.

[1] : Hénon, M. & Heiles, C., The Astronomical Journal **69**, pp 73–79 (1964)
"""
function henonheiles(u0=[0, -0.25, 0.42081, 0]#=; conserveE::Bool = true=#)
    i = one(eltype(u0))
    o = zero(eltype(u0))
    J = zeros(eltype(u0), 4, 4)
    return CDS(hheom!, u0, nothing, hhjacob!, J)
end
function hheom!(du, u, p, t)
    @inbounds begin
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -u[1] - 2u[1]*u[2]
        du[4] = -u[2] - (u[1]^2 - u[2]^2)
        return nothing
    end
end
function hhjacob!(J, u, p, t)
    @inbounds begin
    o = 0.0; i = 1.0
    J[1,:] .= (o,    o,     i,    o)
    J[2,:] .= (o,    o,     o,    i)
    J[3,:] .= (-i - 2*u[2],   -2*u[1],   o,   o)
    J[4,:] .= (-2*u[1],  -1 + 2*u[2],  o,   o)
    return nothing
    end
end
_hhenergy(x,y,px,py) = 0.5(px^2 + py^2) + _hhpotential(x,y)
_hhpotential(x, y) = 0.5(x^2 + y^2) + (x^2*y - (y^3)/3)
function henonheiles_ics(E, n=10)
    ys = range(-0.4, stop = 1.0, length = n)
    pys = range(-0.5, stop = 0.5, length = n)
    ics = Vector{Vector{Float64}}()
    for y in ys
        V = _hhpotential(0.0, y)
        V ≥ E && continue
        for py in pys
            Ky = 0.5*(py^2)
            Ky + V ≥ E && continue
            px = sqrt(2(E - V - Ky))
            ic = [0.0, y, px, py]
            push!(ics, [0.0, y, px, py])
        end
    end
    return ics
end


"""
    qbh([u0]; A=1.0, B=0.55, D=0.4)

```math
\\begin{aligned}
\\dot{q}_0 &= A p_0 \\\\
\\dot{q}_2 &= A p_2 \\\\
\\dot{p}_0 &= -A q_0 -3 \\frac{B}{\\sqrt{2}} (q_2^2 - q_1^2) - D q_1 (q_1^2 + q_2^2) \\\\
\\dot{p}_2 &= -q_2 (A + 3\\sqrt{2} B q_1 + D (q_1^2 + q_2^2)) (x^2 - y^2)
\\end{aligned}
```

These equations of motion correspond to a Hamiltonian used in nuclear
physics to study the quadrupole vibrations of the nuclear surface [1,2].

```math
H(p_0, p_2, q_0, q_2) = \\frac{A}{2}\\left(p_0^2+p_2^2\\right)+\\frac{A}{2}\\left(q_0^2+q_2^2\\right)
			 +\\frac{B}{\\sqrt{2}}q_0\\left(3q_2^2-q_0^2\\right) +\\frac{D}{4}\\left(q_0^2+q_2^2\\right)^2
```

The Hamiltonian has a similar structure with the Henon-Heiles one, but it has an added fourth order term
and presents a nontrivial dependence of chaoticity with the increase of energy [3].
The default initial condition is chaotic.

[1]: Eisenberg, J.M., & Greiner, W., Nuclear theory 2 rev ed. Netherlands: North-Holland pp 80 (1975)

[2]: Baran V. and Raduta A. A., International Journal of Modern Physics E, **7**, pp 527--551 (1998)

[3]: Micluta-Campeanu S., Raportaru M.C., Nicolin A.I., Baran V., Rom. Rep. Phys. **70**, pp 105 (2018)
"""
function qbh(u0=[0., -2.5830294658973876, 1.3873470962626937, -4.743416490252585];  A=1., B=0.55, D=0.4)
    return CDS(qeom, u0, [A, B, D])
end
function qeom(z, p, t)
    @inbounds begin
        A, B, D = p
        p₀, p₂ = z[1], z[2]
        q₀, q₂ = z[3], z[4]

        return SVector{4}(
            -A * q₀ - 3 * B / √2 * (q₂^2 - q₀^2) - D * q₀ * (q₀^2 + q₂^2),
            -q₂ * (A + 3 * √2 * B * q₀ + D * (q₀^2 + q₂^2)),
            A * p₀,
            A * p₂
        )
    end
end

"""
    lorenz96(N::Int, u0 = rand(M); F=0.01)

```math
\\frac{dx_i}{dt} = (x_{i+1}-x_{i-2})x_{i-1} - x_i + F
```

`N` is the chain length, `F` the forcing. Jacobian is created automatically.
(parameter container only contains `F`)
"""
function lorenz96(N::Int, u0 = rand(N); F=0.01)
    @assert N ≥ 4 "`N` must be at least 4"
    lor96 = Lorenz96{N}() # create struct
    return CDS(lor96, u0, [F])
end
struct Lorenz96{N} end # Structure for size type
function (obj::Lorenz96{N})(dx, x, p, t) where {N}
    F = p[1]
    # 3 edge cases
    @inbounds dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + F
    @inbounds dx[2] = (x[3] - x[N]) * x[1] - x[2] + F
    @inbounds dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + F
    # then the general case
    for n in 3:(N - 1)
      @inbounds dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F
    end
    return nothing
end



"""
    duffing(u0 = [rand(), rand(), 0]; ω = 2.2, f = 27.0, d = 0.2, β = 1)
The (forced) duffing oscillator, that satisfies the equation
```math
\\ddot{x} + d\\cdot\\dot{x} + β*x + x^3 = f\\cos(\\omega t)
```
with `f, ω` the forcing strength and frequency and `d` the dampening.

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function duffing(u0 = [rand(), rand()]; ω = 2.2, f = 27.0, d = 0.2, β = 1)

    J = zeros(eltype(u0), 2, 2)
    J[1,2] = 1
    return CDS(duffing_eom, u0, [ω, f, d, β], duffing_jacob)
end
@inbounds function duffing_eom(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector{2, Float64}(dx1, dx2)
end
@inbounds function duffing_jacob(u, p, t)
    ω, f, d, β = p
    return @SMatrix [0 1 ;
    (-β - 3u[1]^2) -d]
end

"""
    shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)
Shinriki oscillator with all other parameters (besides `R1`) set to constants.
*This is a stiff problem, be careful when choosing solvers and tolerances*.
"""
function shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)
    # # Jacobian caller for Shinriki:
    # shinriki_eom(::Type{Val{:jac}}, J, u, p, t) = (shi::Shinriki)(t, u, J)
    return CDS(shinriki_eom, u0, [R1])
end
shinriki_voltage(V) = 2.295e-5*(exp(3.0038*V) - exp(-3.0038*V))
function shinriki_eom(u, p, t)
    R1 = p[1]

    du1 = (1/0.01)*(
    u[1]*(1/6.9 - 1/R1) - shinriki_voltage(u[1] - u[2]) - (u[1] - u[2])/14.5
    )

    du2 = (1/0.1)*(
    shinriki_voltage(u[1] - u[2]) + (u[1] - u[2])/14.5 - u[3]
    )

    du3 = (1/0.32)*(-u[3]*0.1 + u[2])
    return SVector{3}(du1, du2, du3)
end

"""
```julia
gissinger(u0 = 3rand(3); μ = 0.119, ν = 0.1, Γ = 0.9)
```
```math
\\begin{aligned}
\\dot{Q} &= \\mu Q - VD \\\\
\\dot{D} &= -\\nu D + VQ \\\\
\\dot{V} &= \\Gamma -V + QD
\\end{aligned}
```
A continuous system that models chaotic reversals due to Gissinger [1], applied
to study the reversals of the magnetic field of the Earth.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : C. Gissinger, Eur. Phys. J. B **85**, 4, pp 1-12 (2012)
"""
function gissinger(u0 = 3rand(3); μ = 0.119, ν = 0.1, Γ = 0.9)
    return CDS(gissinger_eom, u0, [μ, ν, Γ])
end
function gissinger_eom(u, p, t)
    μ, ν, Γ = p
    du1 = μ*u[1] - u[2]*u[3]
    du2 = -ν*u[2] + u[1]*u[3]
    du3 = Γ - u[3] + u[1]*u[2]
    return SVector{3}(du1, du2, du3)
end

"""
```julia
rikitake(u0 = [1, 0, 0.6]; μ = 1.0, α = 1.0)
```
```math
\\begin{aligned}
\\dot{x} &= -\\mu x +yz \\\\
\\dot{y} &= -\\mu y +x(z-\\alpha) \\\\
\\dot{z} &= 1 - xz
\\end{aligned}
```
Rikitake's dynamo is a system that tries to model the magnetic reversal events
by means of a double-disk dynamo system.

[1] : T. Rikitake Math. Proc. Camb. Phil. Soc. **54**, pp 89–105, (1958)
"""
function rikitake(u0 = [1, 0, 0.6]; μ = 1.0, α = 1.0)
    return CDS(rikitake_eom, u0, [μ, α])
end
function rikitake_eom(u, p, t)
    μ, α = p
    x,y,z = u
    xdot = -μ*x + y*z
    ydot = -μ*y + x*(z - α)
    zdot = 1 - x*y
    return SVector{3}(xdot, ydot, zdot)
end


"""
```julia
nosehoover(u0 = [0, 0.1, 0])
```
```math
\\begin{aligned}
\\dot{x} &= y \\\\
\\dot{y} &= yz - x \\\\
\\dot{z} &= 1 - y^2
\\end{aligned}
```
Three dimensional conservative continuous system, taken from the book
"Elegant Chaos" by J. C. Sprott.
"""
nosehoover(u0 = [0, 0.1, 0]) = CDS(nosehoover_eom, u0, nothing)
function nosehoover_eom(u, p, t)
    x,y,z = u
    xdot = y
    ydot = y*z - x
    zdot  = 1.0 - y*y
    return SVector{3}(xdot, ydot, zdot)
end

"""
```julia
labyrinth(u0 = [1.0, 0, 0])
```
```math
\\begin{aligned}
\\dot{x} &= \\sin(y) \\\\
\\dot{y} &= \\sin(z) \\\\
\\dot{V} &= \\sin(x)
\\end{aligned}
```
Three dimensional conservative continuous system, whose evolution in 3D space looks
like a speudo-random walk, the orbit moving around like in a labyrinth.
Taken from the book "Elegant Chaos" by J. C. Sprott.
"""
labyrinth(u0 = [1.0, 0, 0]) = CDS(labyrinth_eom, u0, nothing)
function labyrinth_eom(u, p, t)
    x,y,z = u
    xdot = sin(y)
    ydot = sin(z)
    zdot = sin(x)
    return SVector{3}(xdot, ydot, zdot)
end


"""
    antidots(u; B = 1.0, d0 = 0.3, c = 0.2)
An antidot "superlattice" is a Hamiltonian system that corresponds to a
smoothened periodic Sinai billiard with disk diameter `d0` and smooth
factor `c` [1].

This version is the two dimensional
classical form of the system, with quadratic equations of motion and
a perpendicular magnetic field. Notice that the equations of motion
are with respect to the velocity instead of momentum, i.e.:
```math
\\begin{aligned}
\\dot{x} &= v_x \\\\
\\dot{y} &= v_y \\\\
\\dot{v_x} &= B*v_y - U_x \\\\
\\dot{v_y} &= -B*v_x - U_X \\\\
\\end{aligned}
```
with ``U`` the potential energy:
```math
U = \\left(\\tfrac{1}{c^4}\\right) \\left[\\tfrac{d_0}{2} + c - r_a\\right]^4
```
if ``r_a = \\sqrt{(x%1)^2 + (y%1)^2} < \\frac{d_0}{2} + c`` and 0
otherwise. I.e. the potential is periodic with period 1 in both ``x, y`` and
normalized such that for energy value of 1 it is a circle of diameter ``d0``.
The magnetic field is also normalized such that for value `B=1` the cyclotron
diameter is 1.

Fo more details see [1].

[1] : G. Datseris *et al*, [arXiv:1711.05833v3](https://arxiv.org/abs/1711.05833v3)
"""
function antidots(u0 = [0.5, 0.5, rand(2)...];
    d0 = 0.5, c = 0.2, B = 1.0)
    return CDS(antidot_eom, u0, [B, d0, c], antidot_jacob)
end

function antidot_eom(u, p, t)
    B, d0, c = p
    x, y, vx, vy = u
    # Calculate quadrant of (x,y):
    U = Uy = Ux = 0.0
    xtilde = x - round(x);  ytilde = y - round(y)
    ρ = sqrt(xtilde^2 + ytilde^2)
    # Calculate derivatives and potential:
    if ρ < 0.5*d0 + c
        sharedfactor = -(4*(c + d0/2 - ρ)^(3))/(c^4*ρ)
        Ux = sharedfactor*xtilde # derivatives
        Uy = sharedfactor*ytilde
    end
    Br = 2*√(2)*B # magnetic field with prefactor
    return SVector{4}(vx, vy, Br*vy - Ux, -vx*Br - Uy)
end
function antidot_jacob(u, p, t)
    B, d0, c = p
    x, y, vx, vy = u
    xtilde = x - round(x);  ytilde = y - round(y)
    ρ = sqrt(xtilde^2 + ytilde^2)
    # Calculate derivatives and potential:
    if ρ < 0.5*d0 + c
        Uxx, Uyy, Uxy = antidot_secondderv(xtilde, ytilde, p)
    else
        Uxx, Uyy, Uxy = 0.0, 0.0, 0.0
    end
    Br = 2*√(2)*B # magnetic field with prefactor
    return SMatrix{4, 4}(0.0, 0, -Uxx, -Uxy, 0, 0, -Uxy, -Uyy,
                         1, 0, 0, -Br,        0, 1, Br, 0)
end

function antidot_potential(x::Real, y::Real, p)
    B, d0, c = p
    δ = 4
    # Calculate quadrant of (x,y):
    xtilde = x - round(x);  ytilde = y - round(y)
    ρ = sqrt(xtilde^2 + ytilde^2)
    # Check distance:
    pot = ρ > 0.5*d0 + c ? 0.0 : (1/(c^δ))*(0.5*d0 + c - ρ)^δ
end

function antidot_secondderv(x, y, p)
    B, d0, c = p
    r = sqrt(x^2 + y^2)
    Uxx = _Uxx(x, y, c, d0, r)
    Uyy = _Uxx(y, x, c, d0, r)
    Uxy =  (2c + d0 - 2r)^2 * (2c + d0 + 4r)*x*y / (2c^4*r^3)
    return Uxx, Uyy, Uxy
end

function _Uxx(x, y, c, d0, r)
    Uxx =  (2c + d0 - 2r)^2 * (-2c*y^2 - d0*y^2 + 2*r*(3x^2 + y^2)) /
    (2 * (c^4) * r^3)
end
