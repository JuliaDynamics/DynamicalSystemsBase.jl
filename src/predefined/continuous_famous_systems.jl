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
The famous three dimensional system due to Lorenz [^Lorenz1963], shown to exhibit
so-called "deterministic nonperiodic flow". It was originally invented to study a
simplified form of atmospheric convection.

Currently, it is most famous for its strange attractor (occuring at the default
parameters), which resembles a butterfly. For the same reason it is
also associated with the term "butterfly effect" (a term which Lorenz himself disliked)
even though the effect applies generally to dynamical systems.
Default values are the ones used in the original paper.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[^Lorenz1963]: E. N. Lorenz, J. atmos. Sci. **20**, pp 130 (1963)
"""
function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CDS(lorenz_rule, u0, [σ, ρ, β])
end
const lorenz63 = lorenz
function lorenz_rule(u, p, t)
    @inbounds begin
        σ = p[1]; ρ = p[2]; β = p[3]
        du1 = σ*(u[2]-u[1])
        du2 = u[1]*(ρ-u[3]) - u[2]
        du3 = u[1]*u[2] - β*u[3]
        return SVector{3}(du1, du2, du3)
    end
end
function lorenz_jacob(u, p, t)
    @inbounds begin
        σ, ρ, β = p
        return SMatrix{3,3}(-σ, ρ - u[3], u[2], σ, -1, u[1], 0, -u[1], -β)
    end
end



"""
```julia
chua(u0 = [0.7, 0.0, 0.0]; a = 15.6, b = 25.58, m0 = -8/7, m1 = -5/7)
```
```math
\\begin{aligned}
\\dot{x} &= a [y - h(x)]\\\\
\\dot{y} &= x - y+z \\\\
\\dot{z} &= b y
\\end{aligned}
```
where ``h(x)`` is defined by
```math
h(x) = m_1 x + \\frac 1 2 (m_0 - m_1)(|x + 1| - |x - 1|)
```
This is a 3D continuous system that exhibits chaos.

Chua designed an electronic circuit with the expressed goal of exhibiting
chaotic motion, and this system is obtained by rescaling the circuit units
to simplify the form of the equation. [^Chua1992]

The parameters are ``a``, ``b``, ``m_0``, and ``m_1``. Setting ``a = 15.6``, ``m_0 = -8/7``
and ``m_1 = -5/7``, and varying the parameter ``b`` from ``b = 25`` to ``b = 51``, one observes
a classic period-doubling bifurcation route to chaos. [^Chua2007]

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[^Chua1992]: Chua, Leon O. "The genesis of Chua's circuit", 1992.

[^Chua2007]:
    [Leon O. Chua (2007) "Chua circuit", Scholarpedia,
    2(10):1488.](http://www.scholarpedia.org/article/Chua_circuit)

"""
function chua(u0 = [0.7, 0.0, 0.0]; a = 15.6, b = 25.58, m0 = -8/7, m1 = -5/7)
    return CDS(chua_rule, u0, [a, b, m0, m1], chua_jacob)
end
function chua_rule(u, p, t)
    @inbounds begin
    a, b, m0, m1 = p
    du1 = a * (u[2] - u[1] - chua_element(u[1], m0, m1))
    du2 = u[1] - u[2] + u[3]
    du3 = -b * u[2]
    return SVector{3}(du1, du2, du3)
    end
end
function chua_jacob(u, p, t)
    a, b, m0, m1 = p
    return @SMatrix[-a*(1 + chua_element_derivative(u[1], m0, m1)) a 0;
                    1 -1 1;
                    0 -b 0]
end
# Helper functions for Chua's circuit.
function chua_element(x, m0, m1)
    return m1 * x + 0.5 * (m0 - m1) * (abs(x + 1.0) - abs(x - 1.0))
end
function chua_element_derivative(x, m0, m1)
    return m1 + 0.5 * (m0 - m1) * (-1 < x < 1 ? 2 : 0)
end



"""
```julia
roessler(u0=[1, -2, 0.1]; a = 0.2, b = 0.2, c = 5.7)
```
```math
\\begin{aligned}
\\dot{x} &= -y-z \\\\
\\dot{y} &= x+ay \\\\
\\dot{z} &= b + z(x-c)
\\end{aligned}
```
This three-dimensional continuous system is due to Rössler [^Rössler1976].
It is a system that by design behaves similarly
to the `lorenz` system and displays a (fractal)
strange attractor. However, it is easier to analyze qualitatively, as for example
the attractor is composed of a single manifold.
Default values are the same as the original paper.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[^Rössler1976]: O. E. Rössler, Phys. Lett. **57A**, pp 397 (1976)
"""
function roessler(u0=[1, -2, 0.1]; a = 0.2, b = 0.2, c = 5.7)
    return CDS(roessler_rule, u0, [a, b, c], roessler_jacob)
end
function roessler_rule(u, p, t)
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
    double_pendulum(u0 = [π/2, 0, 0, 0.5];
                    G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
Famous chaotic double pendulum system (also used for our logo!). Keywords are gravity (`G`),
lengths of each rod (`L1` and `L2`) and mass of each ball (`M1` and `M2`).
Everything is assumed in SI units.

The variables order is ``[θ₁, ω₁, θ₂, ω₂]`` and they satisfy:

```math
\\begin{aligned}
θ̇₁ &= ω₁ \\\\
ω̇₁ &= [M₂ L₁ ω₁² \\sin φ \\cos φ + M₂ G \\sin θ₂ \\cos φ +
       M₂ L₂ ω₂² \\sin φ - (M₁ + M₂) G \\sin θ₁] / (L₁ Δ) \\\\
θ̇₂ &= ω₂ \\\\
ω̇₂ &= [-M₂ L₂ ω₂² \\sin φ \\cos φ + (M₁ + M₂) G \\sin θ₁ \\cos φ -
         (M₁ + M₂) L₁ ω₁² \\sin φ - (M₁ + M₂) G \\sin Θ₂] / (L₂ Δ)
\\end{aligned}
```
where ``φ = θ₂-θ₁`` and ``Δ = (M₁ + M₂) - M₂ \\cos² φ``.

Jacobian is created automatically (thus methods that use the Jacobian will be slower)!

(please contribute the Jacobian in LaTeX :smile:)

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function double_pendulum(u0=[π/2, 0, 0, 0.5];
    G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
    return CDS(doublependulum_rule, u0, [G, L1, L2, M1, M2])
end
@inbounds function doublependulum_rule(u, p, t)
    G, L1, L2, M1, M2 = p

    du1 = u[2]

    φ = u[3] - u[1]
    Δ = (M1 + M2) - M2*cos(φ)*cos(φ)

    du2 = (M2*L1*u[2]*u[2]*sin(φ)*cos(φ) +
               M2*G*sin(u[3])*cos(φ) +
               M2*L2*u[4]*u[4]*sin(φ) -
               (M1 + M2)*G*sin(u[1]))/(L1*Δ)

    du3 = u[4]

    du4 = (-M2*L2*u[4]*u[4]*sin(φ)*cos(φ) +
               (M1 + M2)*G*sin(u[1])*cos(φ) -
               (M1 + M2)*L1*u[2]*u[2]*sin(φ) -
               (M1 + M2)*G*sin(u[3]))/(L2*Δ)

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

The Hénon–Heiles system [^HénonHeiles1964] is a conservative dynamical system and was introduced as a
simplification of the motion of a star around a galactic center. It was originally intended
to study the existence of a "third integral of motion"
(which would make this 4D system integrable).
In that search, the authors encountered chaos, as the third integral existed
for only but a few initial conditions.

The default initial condition is a typical chaotic orbit.
The function `Systems.henonheiles_ics(E, n)` generates a grid of
`n×n` initial conditions, all having the same energy `E`.

[^HénonHeiles1964]: Hénon, M. & Heiles, C., The Astronomical Journal **69**, pp 73–79 (1964)
"""
function henonheiles(u0=[0, -0.25, 0.42081, 0]#=; conserveE::Bool = true=#)
    i = one(eltype(u0))
    o = zero(eltype(u0))
    J = zeros(eltype(u0), 4, 4)
    return CDS(hhrule!, u0, nothing, hhjacob!, J)
end
function hhrule!(du, u, p, t)
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

A conservative dynamical system with rule
```math
\\begin{aligned}
\\dot{q}_0 &= A p_0 \\\\
\\dot{q}_2 &= A p_2 \\\\
\\dot{p}_0 &= -A q_0 -3 \\frac{B}{\\sqrt{2}} (q_2^2 - q_0^2) - D q_0 (q_0^2 + q_2^2) \\\\
\\dot{p}_2 &= -q_2 [A + 3\\sqrt{2} B q_0 + D (q_0^2 + q_2^2)]
\\end{aligned}
```

This dynamical rule corresponds to a Hamiltonian used in nuclear
physics to study the quadrupole vibrations of the nuclear surface [^Eisenberg1975][^Baran1998].

```math
H(p_0, p_2, q_0, q_2) = \\frac{A}{2}\\left(p_0^2+p_2^2\\right)+\\frac{A}{2}\\left(q_0^2+q_2^2\\right)
			 +\\frac{B}{\\sqrt{2}}q_0\\left(3q_2^2-q_0^2\\right) +\\frac{D}{4}\\left(q_0^2+q_2^2\\right)^2
```

The Hamiltonian has a similar structure with the Henon-Heiles one, but it has an added fourth order term
and presents a nontrivial dependence of chaoticity with the increase of energy [^Micluta-Campeanu2018].
The default initial condition is chaotic.

[^Eisenberg1975]:
    Eisenberg, J.M., & Greiner, W., Nuclear theory 2 rev ed.
    Netherlands: North-Holland pp 80 (1975)

[^Baran1998]:
    Baran V. and Raduta A. A., International Journal of Modern Physics E,
    **7**, pp 527--551 (1998)

[^Micluta-Campeanu2018]:
    Micluta-Campeanu S., Raportaru M.C., Nicolin A.I., Baran V., Rom. Rep. Phys.
    **70**, pp 105 (2018)
"""
function qbh(u0=[0., -2.5830294658973876, 1.3873470962626937, -4.743416490252585];  A=1., B=0.55, D=0.4)
    return CDS(qrule, u0, [A, B, D])
end
function qrule(z, p, t)
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
function lorenz96(N::Int, u0 = range(0; length = N, step = 0.1); F=0.01)
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
    duffing(u0 = [0.1, 0.25]; ω = 2.2, f = 27.0, d = 0.2, β = 1)
The (forced) duffing oscillator, that satisfies the equation
```math
\\ddot{x} + d \\dot{x} + β x + x^3 = f \\cos(\\omega t)
```
with `f, ω` the forcing strength and frequency and `d` the damping.

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function duffing(u0 = [0.1, 0.25]; ω = 2.2, f = 27.0, d = 0.2, β = 1)
    J = zeros(eltype(u0), 2, 2)
    J[1,2] = 1
    return CDS(duffing_rule, u0, [ω, f, d, β])
end
@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
end
@inbounds function duffing_jacob(u, p, t)
    ω, f, d, β = p
    return SMatrix{2,2}(0, (-β - 3u[1]^2), 1, -d)
end

"""
    shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)
Shinriki oscillator with all other parameters (besides `R1`) set to constants.
*This is a stiff problem, be careful when choosing solvers and tolerances*.
"""
function shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)
    # # Jacobian caller for Shinriki:
    # shinriki_rule(::Type{Val{:jac}}, J, u, p, t) = (shi::Shinriki)(t, u, J)
    return CDS(shinriki_rule, u0, [R1])
end
shinriki_voltage(V) = 2.295e-5*(exp(3.0038*V) - exp(-3.0038*V))
function shinriki_rule(u, p, t)
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
gissinger(u0 = [3, 0.5, 1.5]; μ = 0.119, ν = 0.1, Γ = 0.9)
```
```math
\\begin{aligned}
\\dot{Q} &= \\mu Q - VD \\\\
\\dot{D} &= -\\nu D + VQ \\\\
\\dot{V} &= \\Gamma -V + QD
\\end{aligned}
```
A continuous system that models chaotic reversals due to Gissinger [^Gissinger2012], applied
to study the reversals of the magnetic field of the Earth.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[^Gissinger2012]: C. Gissinger, Eur. Phys. J. B **85**, 4, pp 1-12 (2012)
"""
function gissinger(u0 = [3, 0.5, 1.5]; μ = 0.119, ν = 0.1, Γ = 0.9)
    return CDS(gissinger_rule, u0, [μ, ν, Γ], gissinger_jacob)
end
function gissinger_rule(u, p, t)
    μ, ν, Γ = p
    du1 = μ*u[1] - u[2]*u[3]
    du2 = -ν*u[2] + u[1]*u[3]
    du3 = Γ - u[3] + u[1]*u[2]
    return SVector{3}(du1, du2, du3)
end
function gissinger_jacob(u, p, t)
    μ, ν, Γ = p
    return @SMatrix [μ -u[3] -u[2];
                     u[3] -ν u[1];
                     u[2] u[1] -1]
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
Rikitake's dynamo [^Rikitake1958] is a system that tries to model the magnetic
reversal events by means of a double-disk dynamo system.

[^Rikitake1958]: T. Rikitake Math. Proc. Camb. Phil. Soc. **54**, pp 89–105, (1958)
"""
function rikitake(u0 = [1, 0, 0.6]; μ = 1.0, α = 1.0)
    return CDS(rikitake_rule, u0, [μ, α], rikitake_jacob)
end
function rikitake_rule(u, p, t)
    μ, α = p
    x,y,z = u
    xdot = -μ*x + y*z
    ydot = -μ*y + x*(z - α)
    zdot = 1 - x*y
    return SVector{3}(xdot, ydot, zdot)
end
function rikitake_jacob(u, p, t)
    μ, α = p
    x,y,z = u
    xdot = -μ*x + y*z
    ydot = -μ*y + x*(z - α)
    zdot = 1 - x*y
    return @SMatrix [-μ  z  y;
                     z-α -μ x;
                     -y  -x 0]
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
Three dimensional conservative continuous system, discovered in 1984 during
investigations in thermodynamical chemistry by Nosé and Hoover, then
rediscovered by Sprott during an exhaustive search as an extremely simple
chaotic system. [^Hoover1995]

See Chapter 4 of "Elegant Chaos" by J. C. Sprott. [^Sprott2010]

[^Hoover1995]:
    Hoover, W. G. (1995). Remark on ‘‘Some simple chaotic flows’’.
    *Physical Review E*, *51*(1), 759.

[^Sprott2010]:
    Sprott, J. C. (2010). *Elegant chaos: algebraically simple chaotic flows*.
    World Scientific.
"""
nosehoover(u0 = [0, 0.1, 0]) = CDS(nosehoover_rule, u0, nothing, nosehoover_jacob)
function nosehoover_rule(u, p, t)
    x,y,z = u
    xdot = y
    ydot = y*z - x
    zdot  = 1.0 - y*y
    return SVector{3}(xdot, ydot, zdot)
end
function nosehoover_jacob(u, p, t)
    x,y,z = u
    return @SMatrix [0 1 0;
                     -1 z y;
                     0 -2y 0]
end

"""
    antidots([u]; B = 1.0, d0 = 0.3, c = 0.2)
An antidot "superlattice" is a Hamiltonian system that corresponds to a
smoothened periodic Sinai billiard with disk diameter `d0` and smooth
factor `c` [^Datseris2019].

This version is the two dimensional
classical form of the system, with quadratic dynamical rule and
a perpendicular magnetic field. Notice that the dynamical rule
is with respect to the velocity instead of momentum, i.e.:
```math
\\begin{aligned}
\\dot{x} &= v_x \\\\
\\dot{y} &= v_y \\\\
\\dot{v_x} &= B v_y - U_x \\\\
\\dot{v_y} &= -B v_x - U_y \\\\
\\end{aligned}
```
with ``U`` the potential energy:
```math
U = \\left(\\tfrac{1}{c^4}\\right) \\left[\\tfrac{d_0}{2} + c - r_a\\right]^4
```
if ``r_a = \\sqrt{(x \\mod 1)^2 + (y \\mod 1)^2} < \\frac{d_0}{2} + c`` and 0
otherwise. That is, the potential is periodic with period 1 in both ``x, y`` and
normalized such that for energy value of 1 it is a circle of diameter ``d_0``.
The magnetic field is also normalized such that for value `B = 1` the cyclotron
diameter is 1.

[^Datseris2019]:
    G. Datseris *et al*,
    [New Journal of Physics 2019](https://iopscience.iop.org/article/10.1088/1367-2630/ab19cc/meta)
"""
function antidots(u0 = [0.5, 0.5, 0.25, 0.25];
    d0 = 0.5, c = 0.2, B = 1.0)
    return CDS(antidot_rule, u0, [B, d0, c], antidot_jacob)
end

function antidot_rule(u, p, t)
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

"""
```julia
ueda(u0 = [3.0, 0]; k = 0.1, B = 12.0)
```
```math
\\ddot{x} + k \\dot{x} + x^3 = B\\cos{t}
```
Nonautonomous Duffing-like forced oscillation system, discovered by Ueda in
1961. It is one of the first chaotic systems to be discovered.

The stroboscopic plot in the (x, ̇x) plane with period 2π creates a "broken-egg
attractor" for k = 0.1 and B = 12. Figure 5 of [^Ruelle1980] is reproduced by

```julia
using Plots
ds = Systems.ueda()
a = trajectory(ds, 2π*5e3, dt = 2π)
scatter(a[:, 1], a[:, 2], markersize = 0.5, title="Ueda attractor")
```

For more forced oscillation systems, see Chapter 2 of "Elegant Chaos" by
J. C. Sprott. [^Sprott2010]

[^Ruelle1980]:
    [Ruelle, David, ‘Strange Attractors’, The Mathematical Intelligencer, 2.3 (1980),
    126–37](https://doi.org/10/dkfd3n)

[^Sprott2010]:
    Sprott, J. C. (2010). *Elegant chaos: algebraically simple chaotic flows*.
    World Scientific.
"""
function ueda(u0 = [3.0, 0]; k = 0.1, B = 12.0)
    return CDS(ueda_rule, u0, [k, B], ueda_jacob)
end
function ueda_rule(u, p, t)
    x,y = u
    k, B = p
    xdot = y
    ydot = B*cos(t) - k*y - x^3
    return SVector{2}(xdot, ydot)
end
function ueda_jacob(u, p, t)
    x,y = u
    k, B = p
    return @SMatrix [0      1;
                     -3*x^2 -k]
end


struct MagneticPendulum
    magnets::Vector{SVector{2, Float64}}
end
mutable struct MagneticPendulumParams
    γs::Vector{Float64}
    d::Float64
    α::Float64
    ω::Float64
end

function (m::MagneticPendulum)(u, p, t)
    x, y, vx, vy = u
    γs::Vector{Float64}, d::Float64, α::Float64, ω::Float64 = p.γs, p.d, p.α, p.ω
    dx, dy = vx, vy
    dvx, dvy = @. -ω^2*(x, y) - α*(vx, vy)
    for (i, ma) in enumerate(m.magnets)
        δx, δy = (x - ma[1]), (y - ma[2])
        D = sqrt(δx^2 + δy^2 + d^2)
        dvx -= γs[i]*(x - ma[1])/D^3
        dvy -= γs[i]*(y - ma[2])/D^3
    end
    return SVector(dx, dy, dvx, dvy)
end

"""
    magnetic_pendulum(u=[0.7,0.7,0,0]; d=0.3, α=0.2, ω=0.5, N=3, γs=fill(1.0,N))

Create a pangetic pendulum with `N` magnetics, equally distributed along the unit circle,
with dynamical rule
```math
\\begin{aligned}
\\ddot{x} &= -\\omega ^2x - \\alpha \\dot{x} - \\sum_{i=1}^N \\frac{\\gamma_i (x - x_i)}{D_i^3} \\\\
\\ddot{y} &= -\\omega ^2y - \\alpha \\dot{y} - \\sum_{i=1}^N \\frac{\\gamma_i (y - y_i)}{D_i^3} \\\\
D_i &= \\sqrt{(x-x_i)^2  + (y-y_i)^2 + d^2}
\\end{aligned}
```
where α is friction, ω is eigenfrequency, d is distance of pendulum from the magnet's plane
and γ is the magnetic strength.
"""
function magnetic_pendulum(u = [sincos(0.12553*2π)..., 0, 0];
    γ = 1.0, d = 0.3, α = 0.2, ω = 0.5, N = 3, γs = fill(γ, N))
    m = MagneticPendulum([SVector(cos(2π*i/N), sin(2π*i/N)) for i in 1:N])
    p = MagneticPendulumParams(γs, d, α, ω)
    return ContinuousDynamicalSystem(m, u, p)
end

"""
    fitzhugh_nagumo(u = 0.5ones(2); a=3.0, b=0.2, ε=0.01, I=0.0)
Famous excitable system which emulates the firing of a neuron, with rule
```math
\\begin{aligned}
\\dot{v} &= av(v-b)(1-v) - w + I \\\\
\\dot{w} &= \\varepsilon(v - w)
\\end{aligned}
```

More details in the [Scholarpedia](http://www.scholarpedia.org/article/FitzHugh-Nagumo_model) entry.
"""
function fitzhugh_nagumo(u = 0.5ones(2); a=3.0, b=0.2, ε=0.01, I=0.0)
    ds = ContinuousDynamicalSystem(fitzhugh_nagumo_rule, u, [a, b, ε, I])
end
function fitzhugh_nagumo_rule(x, p, t)
    u, w = x
    a, b, ε, I = p
    return SVector(a*u*(u-b)*(1. - u) - w + I, ε*(u - w))
end

"""
    more_chaos_example(u = rand(3))
A three dimensional chaotic system introduced in [^Sprott2020] with rule
```math
\\begin{aligned}
\\dot{x} &= y \\\\
\\dot{y} &= -x - \\textrm{sign}(z)y \\\\
\\dot{z} &= y^2 - \\exp(-x^2)
\\end{aligned}
```
It is noteworthy because its strange attractor is multifractal with fractal dimension ≈ 3.

[^Sprott2020]:
    Sprott, J.C. 'Do We Need More Chaos Examples?',
    Chaos Theory and Applications 2(2),1-3, 2020
"""
more_chaos_example(u = [0.0246, 0.79752, 0.3535866]) =
ContinuousDynamicalSystem(more_chaos_rule, u, nothing)
function more_chaos_rule(u, p, t)
    x, y, z = u
    dx = y
    dy = -x - sign(z)*y
    dz = y^2 - exp(-x^2)
    return SVector(dx, dy, dz)
end

"""
    thomas_cyclical(u0 = [1.0, 0, 0]; b = 0.2)
```math
\\begin{aligned}
\\dot{x} &= \\sin(y) - bx\\\\
\\dot{y} &= \\sin(z) - by\\\\
\\dot{z} &= \\sin(x) - bz
\\end{aligned}
```
Thomas' cyclically symmetric attractor is a 3D strange attractor originally proposed
by René Thomas[^Thomas1999]. It has a simple form which is cyclically symmetric in the
x,y, and z variables and can be viewed as the trajectory of a frictionally dampened
particle moving in a 3D lattice of forces.
For more see the [Wikipedia page](https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor).

Reduces to the labyrinth system for `b=0`, see
See discussion in Section 4.4.3 of "Elegant Chaos" by J. C. Sprott.

[^Thomas1999]:
    Thomas, R. (1999). *International Journal of Bifurcation and Chaos*,
    *9*(10), 1889-1905.
"""
thomas_cyclical(u0 = [1.0, 0, 0]; b = 0.2) = ContinuousDynamicalSystem(thomas_rule, u0, [b])
labyrinth(u0 = [1.0, 0, 0]) = ContinuousDynamicalSystem(thomas_rule, u0, [0.0])

function thomas_rule(u, p, t)
    x,y,z = u
    b = p[1]
    xdot = sin(y) - b*x
    ydot = sin(z) - b*y
    zdot = sin(x) - b*z
    return SVector{3}(xdot, ydot, zdot)
end
function thomas_jacob(u, p, t)
    x,y,z = u
    b = p[1]
    return SMatrix{3,3}(-b, 0, cos(x), cos(y), -b, 0, 0, cos(z), -b)
end

"""
    stommel_thermohaline(u = [0.3, 0.2]; η1 = 3.0, η2 = 1, η3 = 0.3)
Stommel's box model for Atlantic thermohaline circulation
```math
\\begin{aligned}
 \\dot{T} &= \\eta_1 - T - |T-S| T \\\\
 \\dot{S} &= \\eta_2 - \\eta_3S - |T-S| S
\\end{aligned}
```
Here ``T, S`` denote the dimensionless temperature and salinity differences respectively
between the boxes (polar and equatorial ocean basins) and ``\\eta_i`` are parameters.

[^Stommel1961]:
    Stommel, Thermohaline convection with two stable regimes of flow. Tellus, 13(2)
"""
function stommel_thermohaline(u = [0.3, 0.2]; η1 = 3.0, η2 = 1, η3 = 0.3)
    ds = ContinuousDynamicalSystem(stommel_thermohaline_rule, u, [η1, η2, η3],
    stommel_thermohaline_jacob)
end
function stommel_thermohaline_rule(x, p, t)
    T, S = x
    η1, η2, η3 = p
    q = abs(T-S)
    return SVector(η1 - T -q*T, η2 - η3*S - q*S)
end
function stommel_thermohaline_jacob(x, p, t)
    T, S = x
    η1, η2, η3 = p
    q = abs(T-S)
    if T ≥ S
        return @SMatrix [(-1 - 2T + S)  (T);
                         (-S)  (-η3 - T + 2S)]
    else
        return @SMatrix [(-1 + 2T - S)  (-T);
                         (+S)  (-η3 + T - 2S)]
    end
end

"""
    lorenz84(u = [0.1, 0.1, 0.1]; F=6.846, G=1.287, a=0.25, b=4.0)
Lorenz-84's low order atmospheric general circulation model
```math
\\begin{aligned}
\\dot x = − y^2 − z^2 − ax + aF, \\\\
\\dot y = xy − y − bxz + G, \\\\
\\dot z = bxy + xz − z. \\\\
\\end{aligned}
```

This system has interesting multistability property in the phase space. For the default
parameter set we have four coexisting attractors that gives birth to interesting fractalized
phase space as shown in [^Freire2008]. One can see this by doing:

```
ds = Systems.lorenz84(rand(3))
xg = yg = range(-1.0, 2.0; length=300)
zg = range(-1.5, 1.5; length=30)
bsn, att = basins_of_attraction((xg, yg, zg), ds; mx_chk_att=4)
```

[^Freire2008]:
    J. G. Freire *et al*,  Multistability, phase diagrams, and intransitivity in the
    Lorenz-84 low-order atmospheric circulation model, Chaos 18, 033121 (2008)
"""
function lorenz84(u = [0.1, 0.1, 0.1]; F=6.846, G=1.287, a=0.25, b=4.0)
    return ContinuousDynamicalSystem(lorenz84_rule, u, [F, G, a, b])
end
@inline @inbounds function lorenz84_rule(u, p, t)
    F, G, a, b = p
    x, y, z = u
    dx = -y^2 -z^2 -a*x + a*F
    dy = x*y - y - b*x*z + G
    dz = b*x*y + x*z - z
    return SVector{3}(dx, dy, dz)
end
function lorenz84_rule_jacob(u, p, t)
    F, G, a, b = p
	x, y, z = u
    return @SMatrix [(-a)  (-2y)  (-2z);
                     y-b*z  x-1  (-b*x);
                     b*y+z  b*x   x-1]
end



"""
    lorenzdl(u = [0.1, 0.1, 0.1]; R=4.7)
Diffusionless Lorenz system: it is *probably* the simplest rotationnaly invariant
chaotic flow.
```math
\\begin{aligned}
\\dot x = y − x, \\\\
\\dot y = -xz, \\\\
\\dot z = xy - R. \\\\
\\end{aligned}
```

For `R=4.7` this system has two coexisting Malasoma strange attractors that are
linked together as shown in [^Sprott2014]. The fractal boundary between the basins
of attractor can be visualized with a Poincaré section at `z=0`:
```julia
ds = Systems.lorenzdl()
xg = yg = range(-10.0, 10.0; length=300)
pmap = poincaremap(ds, (3, 0.), Tmax=1e6; idxs = 1:2)
bsn, att = basins_of_attraction((xg, yg), pmap)
```

[^Sprott2014]:
    J. C. Sprott,  Simplest Chaotic Flows with Involutional Symmetries,
    Int. Jour. Bifurcation and Chaos 24, 1450009 (2014)
"""
function lorenzdl(u = [0.1, 0.1, 0.1]; R=4.7)
    return ContinuousDynamicalSystem(lorenzdl_rule, u, R,
    lorenzdl_rule_jacob)
end
@inline @inbounds function lorenzdl_rule(u, p, t)
    R = p
    x, y, z = u
    dx = y - x
    dy = - x*z
    dz = x*y - R
    return SVector{3}(dx, dy, dz)
end
function lorenzdl_rule_jacob(u, p, t)
    x, y, z = u
    return @SMatrix [-1     1     0;
                     -z     0    -x;
                      y     x     0]
end

"""
    coupled_roessler(u0=[1, -2, 0, 0.11, 0.2, 0.1];
    ω1 = 0.18, ω2 = 0.22, a = 0.2, b = 0.2, c = 5.7, k1 = 0.115, k2 = 0.0)

Two coupled Rössler oscillators, used frequently in the study of chaotic synchronization.
The parameter container has the parameters in the same order as stated in this
function's documentation string.
The equations are:
```math
\\begin{aligned}
\\dot{x_1} &= -\\omega_1 y_1-z_1 \\\\
\\dot{y_1} &= \\omega_1 x+ay_1 + k_1(y_2 - y_1) \\\\
\\dot{z_1} &= b + z_1(x_1-c) \\\\
\\dot{x_2} &= -\\omega_2 y_2-z_2 \\\\
\\dot{y_2} &= \\omega_2 x+ay_2 + k_2(y_1 - y_2) \\\\
\\dot{z_2} &= b + z_2(x_2-c) \\\\
\\end{aligned}
```
"""
function coupled_roessler(u0=[1, -2, 0, 0.11, 0.2, 0.1];
    ω1 = 0.18, ω2 = 0.22, a = 0.2, b = 0.2, c = 5.7, k1 = 0.115, k2 = 0.0)
    p = [ω1, ω2, a, b, c, k1, k2]
    return ContinuousDynamicalSystem(coupled_roessler_f, u0, p)
end
function coupled_roessler_f(u,p,t)
    ω1, ω2, a, b, c, k1, k2 = p

    du1 = -ω1*u[2] - u[3]
    du2 = ω1*u[1] + a*u[2] + k1*(u[5]-u[2])
    du3 = b + u[3]*(u[1]-c)
    du4 = -ω2*u[5] - u[6]
    du5 = ω2*u[4] + a*u[5] + k2*(u[2]-u[5])
    du6 = b + u[6]*(u[4]-c)
    return SVector(du1,du2,du3,du4,du5,du6)
end


"""
    kuramoto(D = 20, u0 = range(0, 2π; length = D);
        K = 0.3, ω = range(-1, 1; length = D)
    )
The Kuramoto model[^Kuramoto1975] of `D` coupled oscillators with equation
```math
\\dot{\\phi}_i = \\omega_i + \\frac{K}{D}\\sum_{j=1}^{D} \\sin(\\phi_j - \\phi_i)
```

[^Kuramoto1975]:
    Kuramoto, Yoshiki.
    International Symposium on Mathematical Problems in Theoretical Physics. 39.
"""
function kuramoto(D = 25, u0 = range(0, 2π; length = D);
    K = 0.3, ω = range(-1, 1; length = D))
    p = KuramotoParams(K, ω)
    @warn "The kuramoto implementation does NOT have a Jacobian function!"
    return ContinuousDynamicalSystem(kuramoto_f, u0, p, (J,z0, p, n) -> nothing)
end
using Statistics: mean
function kuramoto_f(du, u, p, t)
    ω = p.ω; K = p.K
    D = length(u)
    z = mean(exp.(im .* u))
    θ = angle(z)
    @inbounds for i in 1:D
        du[i] = ω[i] + K*abs(z)*sin(θ - u[i])
    end
    return
end
mutable struct KuramotoParams{T<:Real, V<:AbstractVector{T}}
    K::T
    ω::V
end

"""
    sprott_dissipative_conservative(u0 = [1.0, 0, 0]; a = 2, b = 1, c = 1)
An interesting system due to Sprott[^Sprott2014b] where some initial conditios
such as `[1.0, 0, 0]` lead to quasi periodic motion on a 2-torus, while for
`[2.0, 0, 0]` motion happens on a (dissipative) chaotic attractor.

The equations are:
```math
\\begin{aligned}
\\dot{x} &= y + axy + xz \\\\
\\dot{y} &= 1 - 2x^2 + byz \\\\
\\dot{z_1} &= cx - x^2 - y^2
\\end{aligned}
```
In the original paper there were no parameters, which are added here for exploration purposes.

[^Sprott2014b]: J. C. Sprott. Physics Letters A, 378
"""
function sprott_dissipative_conservative(u0 = [1.0, 0, 0]; a = 2, b = 1, c = 1)
    return CDS(
        sprott_dissipative_conservative_f, u0, [a, b, c], sprott_dissipative_conservative_J
    )
end

function sprott_dissipative_conservative_f(u, p, t)
    a, b, c = p
    x, y, z = u
    dx = y + a*x*y + x*z
    dy = 1 - 2*x^2 + b*y*z
    dz = c*x - x^2 - y^2
    return SVector(dx, dy, dz)
end
function sprott_dissipative_conservative_J(u, p, t)
    a, b, c = p
    x, y, z = u
    return @SMatrix [a*y + z     1 + a*x     +x;
                    -4x     b*z    b*y;
                    (c - 2x)    (-2y)  0]
end

"""
```julia
hodgkinhuxley(u0=[-60.0, 0.0, 0.0, 0.0]; I = 12.0, Vna = 50.0, Vk = -77.0, Vl = -54.4, gna = 120.0,gk = 36.0, gl = 0.3) -> ds
```
```math
\\begin{aligned}
C_m \\frac{dV_m}{dt} = -\\overline{g}_\\mathrm{K} n^4 (V_m - V_\\mathrm{K}) - \\overline{g}_\\mathrm{Na} m^3 h(V_m - V_\\mathrm{Na}) - \\overline{g}_l (V_m - Vl) + I\\\\
\\dot{n} &= \\alpha_n(V_m)(1-n) - \\beta_n(V_m)n \\\\
\\dot{m} &= \\alpha_m(V_m)(1-m) - \\beta_m(V_m)m \\\\
\\dot{h} &= \\alpha_h(V_m)(1-h) - \\beta_h(V_m)h \\\\
\\alpha_n(V_m) = \\frac{0.01(V+55)}{1 - \\exp(\\frac{1V+55}{10})} \\quad
\\alpha_m(V_m) = \\frac{0.1(V+40)}{1 - \\exp(\\frac{V+40}{10})} \\quad
\\alpha_h(V_m) = 0.07 \\exp(-\\frac{(V+65)}{20}) \\\\
\\beta_n(V_m) = 0.125 \\exp(-\\frac{V+65}{80}) \\quad
\\beta_m(V_m) = 4 \\exp(-\\frac{V+65}{18}) \\quad
\\beta_h(V_m) = \\frac{1}{1 + \\exp(-\\frac{V+35}{10})}
\\end{aligned}
```
The Nobel-winning four-dimensional dynamical system due to Hodgkin and
Huxley [^HodgkinHuxley1952], which describes the electrical spiking
activity (action potentials) in neurons. A complete description of all parameters and
variables is given in [^HodgkinHuxley1952], [^Ermentrout2010], and [^Abbott2005].
The equations and default parameters used here are taken from [^Ermentrout2010][^Abbott2005].
They differ slightly from the original paper [^HodgkinHuxley1952], since they were
changed to shift the resting potential to -65 mV, instead of the 0mV in the original paper.

Varying the injected current I from `I = -5`  to  `I = 12` takes the neuron from quiescent
to a single spike, and to a tonic (repetitive) spiking. This is due to a subcritical Hopf
bifurcation, which occurs close to `I = 9.5`.

[^HodgkinHuxley1952]: A. L. Hodgkin, A.F. Huxley J. Physiol., pp. 500-544 (1952).

[^Ermentrout2010]:
    G. Bard Ermentrout, and David H. Terman,
    "Mathematical Foundations of Neuroscience", Springer (2010).

[^Abbott2005]:
    L. F. Abbott, and P. Dayan, "Theoretical Neuroscience: Computational and
    Mathematical Modeling of Neural Systems", MIT Press (2005).
"""
function hodgkinhuxley(u0=[-60.0, 0.0, 0.0, 0.0];
    I = 12.0, Vna = 50.0, Vk = -77.0, Vl = -54.4, gna = 120.0,gk = 36.0, gl = 0.3)
#In Ermentrout's & Abbott's books
    return CDS(hodgkinhuxley_rule, u0, [I, Vna, Vk, Vl, gna, gk, gl])
end
function hodgkinhuxley_rule(u, p, t)
    @inbounds begin
        I, Vna, Vk, Vl, gna, gk, gl = p
        V, n, m, h = u

        αn = 0.01 * (V+55)/(1 -  exp(-(V+55)/10))
        αm = 0.1 * (V+40)/(1- exp(-(V+40)/10))
        αh = 0.07 * exp(-(V+65)/20.0 )
        βn = 0.125 * exp(-(V+65)/80.0)
        βm = 4.0 * exp(-(V+65)/18.0 )
        βh = 1.0/(1 + exp(-(V+35)/10))

        du1 = (I -n^4*gk*(V-Vk) - m^3*h*gna*(V-Vna) - gl*(V-Vl))/1.0
        du2 = αn*(1-n) - βn*n
        du3 = αm*(1-m) - βm*m
        du4 = αh*(1-h) - βh*h
        return SVector{4}(du1, du2, du3, du4)
    end
end



"""
```julia
vanderpol(u0=[0.5, 0.0]; μ=1.5, F=1.2, T=10) -> ds
```
```math
\\begin{aligned}
\\ddot{x} -\\mu (1-x^2) \\dot{x} + x = F \\cos(\\frac{2\\pi t}{T})
\\end{aligned}
```
The forced van der Pol oscillator is an oscillator with a nonlinear damping term driven
by a sinusoidal forcing. It was proposed by Balthasar van der Pol, in his studies of
nonlinear electrical circuits used in the first radios [^Kanamaru2007][^Strogatz2015].
The unforced oscillator (`F = 0`) has stable oscillations in the form of a limit cycle
with a slow buildup followed by a sudden discharge, which van der Pol called relaxation
oscillations [^Strogatz2015][^vanderpol1926].
The forced oscillator (`F > 0`) also has periodic behavior for some parameters, but can
additionally have chaotic behavior.

The van der Pol oscillator is a specific case of both the FitzHugh-Nagumo neural
model [^Kanamaru2007]. The default damping parameter is taken from [^Strogatz2015]
and the forcing parameters are taken from [^Kanamaru2007], which generate periodic
oscillations. Setting `\\mu=8.53` generates chaotic oscillations.

[^Kanamaru2007]: Takashi Kanamaru (2007) "Van der Pol oscillator", Scholarpedia, 2(1):2202.

[^Strogatz2015]:
    Steven H. Strogatz (2015) "Nonlinear dynamics and chaos:
    with applications to physics, biology, chemistry, and engineering",
    Boulder, CO :Westview Press, a member of the Perseus Books Group.

[^vanderpol1926]:
    B. Van der Pol (1926), "On relaxation-oscillations",
    The London, Edinburgh and Dublin Phil. Mag. & J. of Sci., 2(7), 978–992.
"""
function vanderpol(u0=[0.5, 0.0]; μ=1.5, F=1.2, T=10)
    return CDS(vanderpol_rule, u0, [μ, F, T], vanderpol_jac)
end
function vanderpol_rule(u, p, t)
    @inbounds begin
        μ, F, T = p
        du1 = u[2]
        du2 = μ*(1 - u[1]^2)*u[2] - u[1] + F*sin(2π*t/T)
        return SVector{2}(du1, du2)
    end
end
function vanderpol_jac(u, p, t)
    @inbounds begin
        μ = p[1]
        J = @SMatrix [0 1;
        (-μ*(2*u[1]-1)*u[2]-1) (μ*(1 - u[1]^2))]
        return J
    end
end

"""
```julia
lotkavolterra(u0=[10.0, 5.0]; α = 1.5, β = 1, δ=1, γ=3) -> ds
```
```math
\\begin{aligned}
\\dot{x} &= \\alpha x - \\beta xy, \\\\
\\dot{y} &= \\delta xy - \\gamma y
\\end{aligned}
```
The famous Lotka-Volterra model is a simple ecological model describing the interaction
between a predator and a prey species (or also parasite and host species).
It has been used independently in fields such as epidemics, ecology,
and economics [^Hoppensteadt2006], and is not to be confused with the
Competitive Lotka-Volterra model, which describes competitive interactions between species.

The `x` variable describes the number of prey, while `y` describes the number of predator.
The default parameters are taken from [^Weisstein], which lead to typical periodic
oscillations.

[^Hoppensteadt2006]:
    Frank Hoppensteadt (2006) "Predator-prey model", Scholarpedia, 1(10):1563.

[^Weisstein]:
    Weisstein, Eric W., "Lotka-Volterra Equations."
    From MathWorld--A Wolfram Web Resource.
    https://mathworld.wolfram.com/Lotka-VolterraEquations.html
"""
function lotkavolterra(u0=[10.0, 5.0]; α = 1.5, β = 1, δ=1, γ=3)
    return CDS(lotkavolterra_rule, u0, [α, β, δ, γ], lotkavolterra_jac)
end
function lotkavolterra_rule(u, p, t)
    @inbounds begin
        α, β, δ, γ = p
        du1 = α*u[1] - β*u[1]*u[2]
        du2 = δ*u[1]*u[2] - γ*u[2]
        return SVector{2}(du1, du2)
    end
end
function lotkavolterra_jac(u, p, t)
    @inbounds begin
        α, β, δ, γ = p
        J = @SMatrix [(α - β*u[2]) (-β*u[1]);
        (δ*u[2]) (δ*u[1] - γ)]
        return J
    end
end

"""
```julia
hindmarshrose(u0=[-1.0, 0.0, 0.0]; a=1, b=3, c=1, d=5, r=0.001, s=4, xr=-8/5, I=2.0) -> ds
```
```math
\\begin{aligned}
\\dot{x} &= y - ax^3 + bx^2 +I - z, \\\\
\\dot{y} &= c - dx^2 -y, \\\\
\\dot{z} &= r(s(x - x_r) - z)
\\end{aligned}
```

The Hindmarsh-Rose model reproduces the bursting behavior of a neuron's membrane potential,
characterized by a fast sequence of spikes followed by a quiescent period.
The `x` variable describes the membane potential, whose behavior can be controlled by
the applied current `I`; the `y` variable describes the sodium and potassium ionic currents,
and `z` describes an adaptation current [^HindmarshRose1984].

The default parameter values are taken from [^HindmarshRose1984], chosen to lead to
periodic bursting.

[^HindmarshRose1984]:
    J. L. Hindmarsh and R. M. Rose (1984)
    "A model of neuronal bursting using three coupled first order differential equations",
    Proc. R. Soc. Lond. B 221, 87-102.
"""
function hindmarshrose(u0=[-1.0, 0.0, 0.0]; a=1, b=3, c=1, d=5, r=0.001, s=4, xr=-8/5, I=2.0)
    return CDS(hindmarshrose_rule, u0, [a,b,c,d,r,s,xr, I], hindmarshrose_jac)
end
function hindmarshrose_rule(u, p, t)
    @inbounds begin
        a,b,c,d,r,s, xr, I = p
        du1 = u[2] - a*u[1]^3 + b*u[1]^2 -u[3] + I
        du2 = c - d*u[1]^2 - u[2]
        du3 = r*(s*(u[1] - xr) - u[3])
        return SVector{3}(du1, du2, du3)
    end
end
function hindmarshrose_jac(u, p, t)
    @inbounds begin
        a,b,c,d,r,s, xr, I = p
        J = @SMatrix [(-3*a*u[1]^2 + 2*b*u[1]) 1 -1;
        -2*d*u[1] -1 0;
        r*s 0 -r]
        return J
    end
end

"""
```julia
hindmarshrose_two_coupled(u0=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
a = 1.0, b = 3.0, d = 5.0, r = 0.001, s = 4.0, xr = -1.6, I = 4.0,
k1 = -0.17, k2 = -0.17, k_el = 0.0, xv = 2.0)
```
```math
\\begin{aligned}
\\dot x_{i} = y_{i} + bx^{2}_{i} - ax^{3}_{i} - z_{i} + I - k_{i}(x_{i} - v_{s})\\Gamma(x_{j}) + k(x_{j} - x_{i})\\\\
\\dot y_{i} = c - d x^{2}_{i} - y_{i}\\\\
\\dot z_{i} = r[s(x_{i} - x_{R}) - z_{i}]\\\\
\\i,j=1,2 (i\\neq j).\\\\
\\end{aligned}
```

The two coupled Hindmarsh Rose element by chemical and electrical synapse.
it is modelling the dynamics of a neuron's membrane potential.
The default parameter values are taken from article "Dragon-king-like extreme events in
coupled bursting neurons", DOI:https://doi.org/10.1103/PhysRevE.97.062311.
"""
function hindmarshrose_two_coupled(u0=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
			a = 1.0, b = 3.0, d = 5.0, r = 0.001, s = 4.0, xr = -1.6, I = 4.0,
			k1 = -0.17, k2 = -0.17, k_el = 0.0, xv = 2.0)
	return CDS(hindmarshrose_coupled_rule, u0, [a, b, c, d, r, s, xr, I, k1, k2, k_el, xv])
end
function hindmarshrose_coupled_rule(u, p, t)
    function sigma(x)
        return @fastmath 1.0 / ( 1.0 + exp( -10.0 * ( x  - ( - 0.25 ) ) ) )
    end
    a, b, c, d, r, s, xr, I, k1, k2, k_el, xv = p
    x1, y1, z1, x2, y2, z2 = u

    du1 = y1 + b * x1 ^ 2 - a * x1 ^3 - z1 + I - k1 * ( x1 - vs ) * sigma(x2) + el_link * ( x2 - x1 )
    du2 = c - d * x1 ^2 - y1
    du3 = r * ( s * ( x1 - xr ) - z1 )

    du4 = y2 + b * x2 ^ 2 - a * x2 ^3 - z2 + I - k2 * ( x2 - vs ) * sigma(x1) + el_link * ( x1 - x2 )
    du5 = c - d * x2 ^2 - y2
    du6 = r * ( s * ( x2 - xr ) - z2 )
    return SVector(du1, du2, du3, du4, du5, du6)
end

"""
```julia
stuartlandau_oscillator(u0=[1.0, 0.0]; μ=1.0, ω=1.0, b=1) -> ds
```
The Stuart-Landau model describes a nonlinear oscillation near a Hopf bifurcation, and was
proposed by Landau in 1944 to explain the transition to turbulence in a fluid [^Landau1944].
It can be written in cartesian coordinates as [^Deco2017]
```math
\\begin{aligned}
\\dot{x} &= (\\mu -x^2 -y^2)x - \\omega y - b(x^2+y^2)y \\\\
\\dot{y} &= (\\mu -x^2 -y^2)y + \\omega x + b(x^2+y^2)x
\\end{aligned}
```

The dynamical analysis of the system is greatly facilitated by putting it in polar coordinates,
where it becomes the normal form of the supercritical Hopf bifurcation) [^Strogatz2015].
```math
\\begin{aligned}
\\dot{r} &= \\mu r - r^3, \\\\
\\dot{\\theta} &= \\omega +br^2
\\end{aligned}
```
The parameter `\\mu` serves as the bifurcation parameter, `\\omega` is the frequency
of infinitesimal oscillations, and `b` controls the dependence of the frequency on
the amplitude.  Increasing `\\mu` from negative to positive generates the supercritical
Hopf bifurcation, leading from a stable spiral at the origin to a stable limit cycle with
radius `\\sqrt(\\mu)`.

[^Landau1944]:
    L. D. Landau, "On the problem of turbulence,
    In Dokl. Akad. Nauk SSSR (Vol. 44, No. 8, pp. 339-349) (1944).
[^Deco2017]:
    G. Deco et al "The dynamics of resting fluctuations in the brain:
    metastability and its dynamical cortical core",  Sci Rep 7, 3095 (2017).
[^Strogatz2015]:
    Steven H. Strogatz "Nonlinear dynamics and chaos:
    with applications to physics, biology, chemistry, and engineering",
    Boulder, CO :Westview Press, a member of the Perseus Books Group (2015).
"""
function stuartlandau_oscillator(u0=[1.0, 0.0]; μ=1.0, ω=1.0, b=1)
    return CDS(stuartlandau_rule, u0, [μ, ω, b], stuartlandau_jac)
end
function stuartlandau_rule(u, p, t)
    @inbounds begin
        μ, ω, b = p
        du1 = u[1]*(μ - u[1]^2 - u[2]^2) - ω*u[2] - b*(u[1]^2 + u[2]^2)*u[2]
        du2 = u[2]*(μ - u[1]^2 - u[2]^2) + ω*u[1] + b*(u[1]^2 + u[2]^2)*u[1]
        return SVector{2}(du1, du2)
    end
end
function stuartlandau_jac(u, p, t)
    @inbounds begin
        μ, ω, b = p
        J = @SMatrix [(μ - 3*u[1]^2 -u[2]^2 -2*b*u[1]*u[2]) (-2*u[1]*u[2] -ω -b*u[1]^2 -3*b*u[2]^2);
            (-2*u[1]*u[2] +ω +b*u[2]^2 +3*b*u[1]^2) (μ -u[1]^2 -3*u[2]^2 +2*b*u[1]*u[2])]
        return J
    end
end


"""
    forced_pendulum(u0 = [0.1, 0.25]; ω = 2.2, f = 27.0, d = 0.2)
The standard forced damped pendulum with a sine response force. duffing oscillator, that satisfies the equation
```math
\\ddot{x} + d \\dot{x} + \\sin(x) = f \\cos(\\omega t)
```
with `f, ω` the forcing strength and frequency and `d` the damping.

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function forced_pendulum(u0 = [0.1, 0.25]; ω = 2.2, f = 27.0, d = 0.2)
    return CDS(forced_pendulum_rule, u0, [ω, f, d])
end
@inbounds function forced_pendulum_rule(u, p, t)
    ω = p[1]; F = p[2]; d = p[3]
    du1 = u[2]
    du2 = -d*u[2] - sin(u[1])+ F*cos(ω*t)
    return SVector{2}(du1, du2)
end

"""
```julia
riddled_basins(u0=[0.5, 0.6, 0, 0]; γ=0.05, x̄ = 1.9, f₀=2.3, ω =3.5, x₀=1, y₀=0) → ds
```
```math
\\begin{aligned}
\\dot{x} &= v_x, \\quad \\dot{y} = v_z \\\\
\\dot{v}_x &= -\\gamma v_x - ( -4x(1-x^2) +y^2) + f_0 \\sin(\\omega t)x_0 \\\\
\\dot{v}_y &= -\\gamma v_y - (2y(x+\\bar{x})) + f_0 \\sin(\\omega t)y_0
\\end{aligned}
```
This 5 dimensional (time-forced) dynamical system was used by Ott et al [^OttRiddled2014]
to analyze *riddled basins of attraction*. This means nearby any point of a basin of attraction
of an attractor A there is a point of the basin of attraction of another attractor B.

[^OttRiddled2014]:
    Ott. et al., [The transition to chaotic attractors with riddled basins](http://yorke.umd.edu/Yorke_papers_most_cited_and_post2000/1994_04_Ott_Alexander_Kan_Sommerer_PhysicaD_riddled%20basins.pdf)
"""
function riddled_basins(u0=[0.5, 0.6, 0, 0];
        γ=0.05, x̄ = 1.9, f₀=2.3, ω =3.5, x₀=1.0, y₀=0.0
    )
    return CDS(riddled_basins_rule, u0, [γ, x̄, f₀, ω, x₀, y₀])
end
function riddled_basins_rule(u, p, t)
    @inbounds begin
        γ, x̄, f₀, ω, x₀, y₀ = p
        x, y, dx, dy = u
        du1 = dx
        du2 = dy
        du3 = -γ*dx -(-4*x*(1-x^2) + y^2) +  f₀*sin(ω*t)*x₀
        du4 = -γ*dy -(2*y*(x+x̄)) +  f₀*sin(ω*t)*y₀
        return SVector{4}(du1, du2, du3, du4)
    end
end

"""
```julia
morris_lecar(u0=[0.1, 0.1]; I = 0.15, V3 = 0.1, V1 = -0.00, V2 = 0.15, V4 = 0.1,
    VCa = 1, VL = -0.5, VK = -0.7, gCa = 1.2, gK = 2, gL = 0.5, τ = 3) -> ds
```
The Morris-Lecar model is ubiquitously used in computational neuroscience as a
**simplified model for neuronal dynamics** (2D), and can also be in general as an
excitable system [^IzhikevichBook]. It uses the formalism of the more complete
Hodgkin-Huxley model (4D), and can be viewed as a simplification thereof, with variables V
for the membrane potential and N for the recovery of the Potassium current. Its original
parameters were obtained from experimental studies of the giant muscle fiber in the
Pacific barnacle [^MorrisLecar1981]. Its evolution is given by:
```math
\\begin{aligned}
\\dot{V} &= -g_{Ca} M(V) (V - V_{Ca}) - g_K N (V - V_K) - g_L (V - V_L) + I \\\\
\\dot{N} &= (1/\tau) (-N + G(V)) \\\\
\\end{aligned}
```
with
```math
\\begin{aligned}
M(V) = 0.5 (1 + \tanh((x-V1)/V2)) \\\\
G(V) = 0.5 (1 + \tanh((x-V3)/V4)) \\\\
```

[^IzhikevichBook]:
    Izhikevich, E. M., Dynamical systems in neuroscience:
    The geometry of excitability and bursting,
    2007, MIT Press.
[^MorrisLecar1981]:
    Morris, C. and Lecar, H, [Voltage oscillations in the barnacle giant muscle fiber,
    1981](https://www.cell.com/biophysj/pdf/S0006-3495(81)84782-0.pdf).
"""
function morris_lecar(u0=[0.1, 0.1];
    I = 0.15, V3 = 0.1, V1 = -0.00, V2 = 0.15, V4 = 0.1,
    VCa = 1 ,  VL = -0.5, VK = -0.7, gCa = 1.2, gK = 2,
    gL = 0.5, τ = 3)
    return CDS(morris_lecar_rule, u0, [I, V3, V1, V2, V4, VCa, VL, VK, gCa, gK, gL, τ])
end

@inbounds function morris_lecar_rule(u, p, t)
    I, V3, V1, V2, V4, VCa, VL, VK, gCa, gK, gL, τ = p
    V, N = u
    M = 0.5*(1 + tanh((V-V1)/V2))
    G = 0.5*(1 + tanh((V-V3)/V4))
    du1 = -gCa*M*(V - VCa) -gK*N*(V - VK) -gL*(V-VL) + I
    du2 = 1/τ*(-N + G)
    return SVector{2}(du1, du2)
end

"""
```julia
sakarya(u0= [-2.8976045, 3.8877978, 3.07465];
    a = 1,
    b = 1,
    m = 1
)
```
```math
\\begin{aligned}
\\dot{x} &= ax + y + yz\\\\
\\dot{y} &= - xz + yz \\\\
\\dot{z} &= - z - mxy + b
\\end{aligned}
```
A system presenting robust chaos that varies from single wing to double wings to four wings.
Its attractor arises due to merging of two disjoint bistable attractors [^Li2015].

[^Li2015]:
    Li, Chunbiao, et al (2015). A novel four-wing strange attractor born in bistability.
    IEICE Electronics Express 12.4.
"""
function sakarya(u0= [-2.8976045, 3.8877978, 3.07465];
    a = 1,
    b = 1,
    m = 1
)
    return CDS(sakarya_rule, u0, [a,b,m])
end

function sakarya_rule(u, p, t)
    @inbounds begin
        a, b, m = p
        du1 = a*u[1] + u[2] + u[2]*u[3]
        du2 = - u[1]*u[3] + u[2]*u[3]
        du3 = - u[3] - m*u[1]*u[2] + b
    end
    return SVector{3}(du1, du2, du3)
end


"""
```julia
lorenz_bounded(u0=[-13.284881, -12.444334, 34.188198];
    beta = 2.667,
    r = 64.0,
    rho = 28.0,
    sigma = 10.0
)
```
```math
\\begin{aligned}
\\dot{X} &= \\sigma(Y-X)f(X,Y,Z) \\\\
\\dot{Y} &= (-XZ + \\rho X -Y)f(X,Y,Z) \\\\
\\dot{Z} &= (XY - \\beta Z)f(X,Y,Z)
\\end{aligned}
```

Lorenz system bounded by a confining potential [^SprottXiong2015].

[^SprottXiong2015]:
    Sprott, J. C., & Xiong, A. (2015). Classifying and quantifying basins of attraction.
    Chaos: An Interdisciplinary Journal of Nonlinear Science, 25(8), 083101.
"""
function lorenz_bounded(u0=[-13.284881, -12.444334, 34.188198];
    beta = 2.667,
    r = 64.0,
    rho = 28.0,
    sigma = 10.0
)
    return CDS(lorenzbounded_rule, u0, [beta,r,rho,sigma])
end

function lorenzbounded_rule(u, p, t)
    @inbounds begin
        beta, r, rho, sigma = p
        f = 1 - (u[1]^2 + u[2]^2 + u[3]^2) / r^2
        du1 = sigma * (u[2] - u[1])*f
        du2 = (u[1]*(rho - u[3]) - u[2])*f
        du3 = (u[1]*u[2] - beta*u[3])*f
    end
    return SVector{3}(du1, du2, du3)
end

"""
```julia
swinging_atwood(u0=[0.113296,1.5707963267948966,0.10992,0.17747]; m1=1.0, m2=4.5)
```
```math
\\begin{aligned}
\\dot{r} &= \\frac{p_r}{M+m}\\\\
\\dot{p}_r &= -Mg + mg\\cos(\\theta)\\\\
\\dot{\\theta} &= \\frac{p_{\\theta}}{mr^2}\\\\
\\dot{p}_{\\theta} &= -mgr\\sin(\\theta)
\\end{aligned}
```
A mechanical system consisting of two swinging weights connected by ropes and pulleys.
This is only chaotic when `m2` is sufficiently larger than `m1`, and there are nonzero initial
momenta [^Tufillaro1984].

[^Tufillaro1984]:
    Tufillaro, Nicholas B.; Abbott, Tyler A.; Griffiths, David J. (1984).
    Swinging Atwood's Machine. American Journal of Physics. 52 (10): 895--903.
"""
function swinging_atwood(u0=[0.113296,1.5707963267948966,0.10992,0.17747]; m1=1.0, m2=4.5)
    return CDS(swingingatwood_rule, u0, [m1, m2])
end

function swingingatwood_rule(u, p, t)
    @inbounds begin
        m1, m2 = p
        r, th, pr, pth = u
        g = 9.82
        du1 = pr/(m1 + m2)
        du2 = pth/(m1*r^2)
        du3 = pth^2/(m1*r^3) - m2*g + m1*g*cos(th)
        du4 = -m1*g*r*sin(th)
    end
    return SVector{4}(du1, du2, du3, du4)
end

"""
```julia
guckenheimer_holmes(u0=[-0.55582369,0.05181624,0.37766104];
    a = 0.4,
    b = 20.25,
    c = 3,
    d = 1.6,
    e = 1.7,
    f = 0.44)
```
```math
\\begin{aligned}
\\dot{x} &= ax - by + czx + dz(x^2 + y^2)\\\\
\\dot{y} &= ay + bx + czy\\\\
\\dot{z} &= e - z^2 - f(x^2 + y^2) - az^3
\\end{aligned}
```
A nonlinear oscillator [^GuckenheimerHolmes1983].

[^GuckenheimerHolmes1983]:
    Guckenheimer, John, and Philip Holmes (1983).
    Nonlinear oscillations, dynamical systems, and bifurcations of vector fields.
    Vol. 42. Springer Science & Business Media.
"""
function guckenheimer_holmes(u0=[-0.55582369,0.05181624,0.37766104];
    a = 0.4,
    b = 20.25,
    c = 3,
    d = 1.6,
    e = 1.7,
    f = 0.44)
    return CDS(guckenheimerholmes_rule, u0, [a,b,c,d,e,f])
end

function guckenheimerholmes_rule(u, p, t)
    @inbounds begin
        a,b,c,d,e,f = p
        x,y,z = u
        du1 = a*x - b*y + c*z*x + d*z*(x^2 + y^2)
        du2 = a*y + b*x + c*z*y
        du3 = e - z^2 - f*(x^2 + y^2) - a*z^3
    end
    return SVector{3}(du1, du2, du3)
end

"""
```julia
halvorsen(u0=[-8.6807408,-2.4741399,0.070775762]; a = 1.4, b = 4.0)
```
```math
\\begin{aligned}
\\dot{x} &= -a*x - b*(y + z) - y^2\\\\
\\dot{y} &= -a*y - b*(z + x) - z^2\\\\
\\dot{z} &= -a*z - b*(x + y) - x^2
\\end{aligned}
```
An algebraically-simple chaotic system with quadratic nonlinearity [^Sprott2010].

[^Sprott2010]:
    Sprott, Julien C (2010). Elegant chaos: algebraically simple chaotic flows.
    World Scientific, 2010.
"""
function halvorsen(u0=[-8.6807408,-2.4741399,0.070775762]; a = 1.4, b = 4.0)
    return CDS(halvorsen_rule, u0, [a, b])
end

function halvorsen_rule(u, p, t)
    @inbounds begin
        x, y, z = u
        a, b = p
        du1 = -a*x - b*(y + z) - y^2
        du2 = -a*y - b*(z + x) - z^2
        du3 = -a*z - b*(x + y) - x^2
    end
    return SVector{3}(du1, du2, du3)
end

"""
    multispecies_competition(option = 1)
A model of competition dynamics between multiple species from Huisman and Weissing[^Huisman2001].
It highlights fundamental unpredictability by having extreme multistability,
fractal basin boundaries and transient chaos.

# TODO: write here equations when we have access to the paper (not open access).
# TODO: Describe initial conditions and what option 1 means.

[^Huisman2001]:
    Huisman & Weissing 2001, Fundamental Unpredictability in Multispecies Competition
    [The American Naturalist Vol. 157, No. 5.](https://doi.org/10.1086/319929)
"""
function multispecies_competition(option = 1)
    p = CompetitionDynamicsParameters(option)
    N = size(p.Ks, 2)
    u0 = [[0.1 for i=1:N]; [S for S in p.Ss]]
    ds = ContinuousDynamicalSystem(multispecies_competition_rule!, u0, p, (J, x, p, t) -> nothing)
    return ds
end

multispecies_competition_monod(r, R, K) = r*R/(K+R)
function _mc_μ!(μs, rs, Rs, Ks)
    for i in eachindex(μs)
        mo1 = multispecies_competition_monod(rs[i], Rs[1], Ks[1,i])
        mo2 = multispecies_competition_monod(rs[i], Rs[2], Ks[2,i])
        mo3 = multispecies_competition_monod(rs[i], Rs[3], Ks[3,i])
        μs[i] = min(mo1, mo2, mo3)
    end
    return nothing
end

#not the most optimized but runs fine
function _mc_Rcoup!(Rcoups, Ns, Rs, μs, cs)
    fill!(Rcoups, 0.0)
    for j in eachindex(Rcoups)
        for i in eachindex(μs)
            Rcoups[j] += cs[j,i] * μs[i] * Ns[i]
        end
    end
    return nothing
end

function multispecies_competition_rule!(du, u, p, t)
    (; rs, Ks, ms, Ss, cs, μs, Rcoups, D) = p
    n = size(Ks, 2)
    Ns = view(u, 1:n)
    Rs = view(u, n+1:n+3)
    dNs = view(du, 1:n)
    dRs = view(du, n+1:n+3)
    _mc_μ!(μs, rs, Rs, Ks)
    _mc_Rcoup!(Rcoups, Ns, Rs, μs, cs)
    @. dNs = Ns * (μs - ms)
    @. dRs = D*(Ss - Rs) - Rcoups
    return nothing
end

mutable struct CompetitionDynamicsParameters
    rs :: Vector{Float64}
    ms :: Vector{Float64}
    Ss :: Vector{Float64}
    μs :: Vector{Float64}
    Rcoups :: Vector{Float64}
    Ks :: Matrix{Float64}
    cs :: Matrix{Float64}
    D :: Float64
end

function CompetitionDynamicsParameters(option = 1)
    if option == 2
        Ks  = [
            0.20 0.05 0.50 0.05 0.50 0.03 0.51 0.51;
            0.15 0.06 0.05 0.50 0.30 0.18 0.04 0.31;
            0.15 0.50 0.30 0.06 0.05 0.18 0.31 0.04;
        ]
        cs = [
            0.20 0.10 0.10 0.10 0.10 0.22 0.10 0.10;
            0.10 0.20 0.10 0.10 0.20 0.10 0.22 0.10;
            0.10 0.10 0.20 0.20 0.10 0.10 0.10 0.22;
        ]
    elseif option == 1
        Ks = [
            0.20 0.05 1.00 0.05 1.20;
            0.25 0.10 0.05 1.00 0.40;
            0.15 0.95 0.35 0.10 0.05;
        ]
        cs = [
            0.20 0.10 0.10 0.10 0.10;
            0.10 0.20 0.10 0.10 0.20;
            0.10 0.10 0.20 0.20 0.10;
        ]
    end
    N = size(Ks, 2)
    rs = [1.0 for i=1:N]
    D = 0.25
    ms = [D for i=1:N]
    Ss = [10.0 for j=1:3]
    μs = zeros(Float64, N)
    Rcoups = zeros(Float64, 3)
    return CompetitionDynamicsParameters(rs, ms, Ss, μs, Rcoups, Ks, cs, D)
end