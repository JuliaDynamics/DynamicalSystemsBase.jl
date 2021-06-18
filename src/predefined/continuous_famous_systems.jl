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
chua(u0 = [0.7, 0.0, 0.0]; a = 15.6, b = 25.58, m0 = -8/7, m1 = -5/7)
```
```math
\\begin{aligned}
\\dot{x} &= \\alpha (y - h(x))\\\\
\\dot{y} &= x - y+z \\\\
\\dot{z} &= \\beta y
\\end{aligned}
```
where h(x) is defined by
```math
h(x) = m_1 x + \\frac 1 2 (m_0 - m_1)(|x + 1| - |x - 1|)
```
This is a 3D continuous system that exhibits chaos.

Chua designed an electronic circuit with the expressed goal of exhibiting
chaotic motion, and this system is obtained by rescaling the circuit units
to simplify the form of the equation. [1]

The parameters are a, b, m0 and m1. Setting a = 15.6, m0 = -8/7 and m1 = -5/7,
and varying the parameter b from b = 25 to b = 51, one observes a classic
period-doubling bifurcation route to chaos. [2]

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : Chua, Leon O. "The genesis of Chua's circuit", 1992.

[2] : [Leon O. Chua (2007) "Chua circuit", Scholarpedia, 2(10):1488.](http://www.scholarpedia.org/article/Chua_circuit)

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
Famous chaotic double pendulum system (also used for our logo!). Keywords
are gravity (G), lengths of each rod (L₁ and L₂) and mass of each ball (M₁ and M₂).
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
where ``φ = θ₂-θ₁``, ``Δ = (M₁ + M₂) - M₂ \\cos² φ``.

Jacobian is created automatically (thus methods that use the Jacobian will be slower)!

(please contribute the Jacobian in LaTeX :smile:)

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function double_pendulum(u0=[π/2, 0, 0, 0.5]; G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
    return CDS(doublependulum_rule, u0, [G, L1, L2, M1, M2])
end
@inbounds function doublependulum_rule(state, p, t)
    G, L1, L2, M1, M2 = p

    du1 = state[2]

    φ = state[3] - state[1]
    Δ = (M1 + M2) - M2*cos(φ)*cos(φ)
    
    du2 = (M2*L1*state[2]*state[2]*sin(φ)*cos(φ) +
               M2*G*sin(state[3])*cos(φ) +
               M2*L2*state[4]*state[4]*sin(φ) -
               (M1 + M2)*G*sin(state[1]))/(L1*Δ)

    du3 = state[4]

    du4 = (-M2*L2*state[4]*state[4]*sin(φ)*cos(φ) +
               (M1 + M2)*G*sin(state[1])*cos(φ) -
               (M1 + M2)*L1*state[2]*state[2]*sin(φ) -
               (M1 + M2)*G*sin(state[3]))/(L2*Δ)
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

The Hénon–Heiles system [1] is a conservative dynamical system and was introduced as a simplification of the motion
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
\\dot{p}_0 &= -A q_0 -3 \\frac{B}{\\sqrt{2}} (q_2^2 - q_1^2) - D q_1 (q_1^2 + q_2^2) \\\\
\\dot{p}_2 &= -q_2 [A + 3\\sqrt{2} B q_1 + D (q_1^2 + q_2^2)] (x^2 - y^2)
\\end{aligned}
```

This dynamical rule corresponds to a Hamiltonian used in nuclear
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
    return CDS(duffing_rule, u0, [ω, f, d, β], duffing_jacob, J)
end
@inbounds function duffing_rule(x, p, t)
    ω, f, d, β = p
    dx1 = x[2]
    dx2 = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
    return SVector(dx1, dx2)
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
A continuous system that models chaotic reversals due to Gissinger [1], applied
to study the reversals of the magnetic field of the Earth.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : C. Gissinger, Eur. Phys. J. B **85**, 4, pp 1-12 (2012)
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
Rikitake's dynamo is a system that tries to model the magnetic reversal events
by means of a double-disk dynamo system.

[1] : T. Rikitake Math. Proc. Camb. Phil. Soc. **54**, pp 89–105, (1958)
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
chaotic system. [1]

See Chapter 4 of "Elegant Chaos" by J. C. Sprott. [2]

[1] : Hoover, W. G. (1995). Remark on ‘‘Some simple chaotic flows’’. *Physical Review E*, *51*(1), 759.

[2] : Sprott, J. C. (2010). *Elegant chaos: algebraically simple chaotic flows*. World Scientific.
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
                     0 2y 0]
end

"""
    antidots([u]; B = 1.0, d0 = 0.3, c = 0.2)
An antidot "superlattice" is a Hamiltonian system that corresponds to a
smoothened periodic Sinai billiard with disk diameter `d0` and smooth
factor `c` [1].

This version is the two dimensional
classical form of the system, with quadratic dynamical rule and
a perpendicular magnetic field. Notice that the dynamical rule
is with respect to the velocity instead of momentum, i.e.:
```math
\\begin{aligned}
\\dot{x} &= v_x \\\\
\\dot{y} &= v_y \\\\
\\dot{v_x} &= B v_y - U_x \\\\
\\dot{v_y} &= -B v_x - U_X \\\\
\\end{aligned}
```
with ``U`` the potential energy:
```math
U = \\left(\\tfrac{1}{c^4}\\right) \\left[\\tfrac{d_0}{2} + c - r_a\\right]^4
```
if ``r_a = \\sqrt{(x \\mod 1)^2 + (y \\mod 1)^2} < \\frac{d_0}{2} + c`` and 0
otherwise. I.e. the potential is periodic with period 1 in both ``x, y`` and
normalized such that for energy value of 1 it is a circle of diameter ``d0``.
The magnetic field is also normalized such that for value `B=1` the cyclotron
diameter is 1.

[1] : G. Datseris *et al*, [New Journal of Physics 2019](https://iopscience.iop.org/article/10.1088/1367-2630/ab19cc/meta)
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
attractor" for k = 0.1 and B = 12. Figure 5 of [1] is reproduced by

```julia
using Plots
ds = Systems.ueda()
a = trajectory(ds, 2π*5e3, dt = 2π)
scatter(a[:, 1], a[:, 2], markersize = 0.5, title="Ueda attractor")
```

For more forced oscillation systems, see Chapter 2 of "Elegant Chaos" by
J. C. Sprott. [2]

[1] : [Ruelle, David, ‘Strange Attractors’, The Mathematical Intelligencer, 2.3 (1980), 126–37](https://doi.org/10/dkfd3n)

[2] : Sprott, J. C. (2010). *Elegant chaos: algebraically simple chaotic flows*. World Scientific.
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
    ds = ContinuousDynamicalSystem(m, u, p)
end

"""
    fitzhugh_nagumo(u = 0.5ones(2); a=3.0, b=0.2, ε=0.01, I=0.0)
Famous excitable system which emulates the firing of a neuron, with rule
```math
\\begin{aligned}
\\dot{v} &= av(v-b)(1-v) - w + I \\\\
\\ddot{w} &= \\varepsilon(v - w)
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

[^Sprott2020]: Sprott, J.C. 'Do We Need More Chaos Examples?', Chaos Theory and Applications 2(2),1-3, 2020
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

[^Thomas1999]: Thomas, R. (1999). *International Journal of Bifurcation and Chaos*, *9*(10), 1889-1905.
"""
thomas_cyclical(u0 = [1.0, 0, 0]; b = 0.2) = CDS(thomas_rule, u0, [b], thomas_jacob)
labyrinth(u0 = [1.0, 0, 0]) = CDS(thomas_rule, u0, [0.0], thomas_jacob)

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
    return @SMatrix [b cos(y) 0;
                     0 b cos(z);
                     cos(x) 0 b]
end

"""
    stommel_thermohaline(u = [0.3, 0.2]; η1 = 3.0, η2 = 1, η3 = 0.3)
Stommel's box model for Atlantic thermohaline circulation
```math
\\begin{aligned}
 \\dot{T} &= \\eta_1 - T - |T-S|\\cdot T \\\\
 \\dot{S} &= \\eta_2 - \\eta_3S - |T-S|\\cdot S
\\end{aligned}
```
Here ``T, S`` are variables standing for dimensionless temperature and salinity
differences between the boxes (polar and equitorial ocean basins) and ``\\eta_i``
are parameters.

[^Stommel1961]: Stommel, Thermohaline convection with two stable regimes offlow. Tellus, 13(2)
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
