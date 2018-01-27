"""
Sub-module of the module `DynamicalSystemsBase`, which contains pre-defined
famous systems.
"""
module Systems
using DynamicalSystemsBase, StaticArrays
using DiffEqCallbacks
const twopi = 2π
#######################################################################################
#                                    Continuous                                       #
#######################################################################################
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

    J = zeros(eltype(u0), 3, 3)
    J[1,:] .= (-σ,          σ,    0)
    J[2,:] .= (ρ - u0[3],  -1,   -u0[1])
    J[3,:] .= (u0[2],   u0[1],   -β)

    @inline @inbounds function lorenz63_eom(du, u, p, t)
        σ, ρ, β = p
        du[1] = σ*(u[2]-u[1])
        du[2] = u[1]*(ρ-u[3]) - u[2]
        du[3] = u[1]*u[2] - β*u[3]
        return nothing
    end

    @inline @inbounds function lorenz63_jacob(J, u, p, t)
        σ, ρ, β = p
        J[1,1] = -σ; J[1, 2] = σ
        J[2,1] = ρ - u[3]; J[2,3] = -u[1]
        J[3,1] = u[2]; J[3,2] = u[1]; J[3,3] = -β
        return nothing
    end

    return ContinuousDS(u0, lorenz63_eom, lorenz63_jacob, J; parameters = [σ, ρ, β])
end



"""
```julia
roessler(u0=rand(3); a = 0.2, b = 0.2, c = 5.7)
```
```math
\\begin{aligned}
\\dot{x} &= -y-z \\\\
\\dot{y} &= x+ay \\\\
\\dot{z} &= -b + z(x-c)
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
    i = one(eltype(u0))
    o = zero(eltype(u0))
    J = zeros(eltype(u0), 3, 3)
    J[1,:] .= [o, -i,      -i]
    J[2,:] .= [i,  a,       o]
    J[3,:] .= [u0[3], o, u0[1] - c]

    @inline @inbounds function roessler_eom(du, u, p, t)
        a, b, c = p
        du[1] = -u[2]-u[3]
        du[2] = u[1] + a*u[2]
        du[3] = b + u[3]*(u[1] - c)
        return nothing
    end

    @inline @inbounds function roessler_jacob(J, u, p, t)
        J[2,2] = p[1]
        J[3,1] = u[3]; J[3,3] = u[1] - p[3]
        return nothing
    end

    return ContinuousDS(u0, roessler_eom, roessler_jacob, J; parameters = [a, b, c])
end


"""
    double_pendulum(u0=rand(4); G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)
Famous chaotic double pendulum system (also used for our logo!). Keywords
are gravity (G), lengths of each rod and mass of each ball (all assumed SI units).

The variables order is [θ1, dθ1/dt, θ2, dθ2/dt].

Jacobian is created automatically (thus methods that use the Jacobian will be slower)!

(please contribute the Jacobian and the e.o.m. in LaTeX :smile:)

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function double_pendulum(u0=rand(4); G=10.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)

    @inbounds function doublependulum_eom(du, state, p, t)
        G, L1, L2, M1, M2 = p

        du[1] = state[2]
        del_ = state[3] - state[1]
        den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
        du[2] = (M2*L1*state[2]*state[2]*sin(del_)*cos(del_) +
                   M2*G*sin(state[3])*cos(del_) +
                   M2*L2*state[4]*state[4]*sin(del_) -
                   (M1 + M2)*G*sin(state[1]))/den1

        du[3] = state[4]

        den2 = (L2/L1)*den1
        du[4] = (-M2*L2*state[4]*state[4]*sin(del_)*cos(del_) +
                   (M1 + M2)*G*sin(state[1])*cos(del_) -
                   (M1 + M2)*L1*state[2]*state[2]*sin(del_) -
                   (M1 + M2)*G*sin(state[3]))/den2
        return nothing
    end
    return ContinuousDS(u0, doublependulum_eom; parameters = [G, L1, L2, M1, M2])
end


"""
    henonhelies(u0=[0, -0.25, 0.42081,0]; conserveE = true)
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

You can optionally choose to conserve energy, up to `1e-14` error level, but having
slower integration as a drawback.

[1] : Hénon, M. & Heiles, C., The Astronomical Journal **69**, pp 73–79 (1964)
"""
function henonhelies(u0=[0, -0.25, 0.42081, 0]; conserveE::Bool = true)

    function hheom!(du, u, p, t)
        du[1] = u[3]
        du[2] = u[4]
        du[3] = -u[1] - 2u[1]*u[2]
        du[4] = -u[2] - (u[1]^2 - u[2]^2)
        return nothing
    end
    function hhjacob!(J, u, p, t)
        J[3,1] = -1 - 2u[2]; J[3,2] = -2u[1]
        J[4,1] = -2u[1]; J[4,2] =  -1 + 2u[2]
        return nothing
    end

    i = one(eltype(u0))
    o = zero(eltype(u0))
    J = zeros(eltype(u0), 4, 4)
    J[1,:] = [o,    o,     i,    o]
    J[2,:] = [o,    o,     o,    i]
    J[3,:] = [ -i - 2*u0[2],   -2*u0[1],   o,   o]
    J[4,:] = [-2*u0[1],  -1 + 2*u0[2],  o,   o]

    @inline Vhh(q1, q2) = 1//2 * (q1^2 + q2^2 + 2q1^2 * q2 - 2//3 * q2^3)
    @inline Thh(p1, p2) = 1//2 * (p1^2 + p2^2)
    @inline Hhh(q1, q2, p1, p2) = Thh(p1, p2) + Vhh(q1, q2)
    @inline Hhh(u::AbstractVector) = Hhh(u...)

    E = Hhh(u0)

    function ghh!(resid, u)
        resid[1] = Hhh(u[1],u[2],u[3],u[4]) - E
        resid[2:4] .= 0
    end


    if conserveE
        cb = ManifoldProjection(ghh!, nlopts=Dict(:ftol=>1e-13), save = false)
        prob = ODEProblem(hheom!, u0, (0., 100.0),  callback=cb)
    else
        prob = prob = ODEProblem(hheom!, u0, (0., 100.0))
    end
    return ContinuousDS(prob, hhjacob!, J)
end


"""
    lorenz96(N::Int, u0 = rand(M); F=0.01)
`N` is the chain length, `F` the forcing. Jacobian is created automatically.
(parameter container only contains `F`)
"""
function lorenz96(N::Int, u0 = rand(N); F=0.01)
    @assert N ≥ 3 "`N` must be at least 3"
    lor96 = Lorenz96{N}() # create struct
    return ContinuousDS(u0, lor96; parameters = [F])
end
struct Lorenz96{N} end # Structure for size type
function (obj::Lorenz96{N})(dx, x, p, t) where {N}
    F = p[1]
    # 3 edge cases
    dx[1] = (x[2] - x[N - 1]) * x[N] - x[1] + F
    dx[2] = (x[3] - x[N]) * x[1] - x[2] + F
    dx[N] = (x[1] - x[N - 2]) * x[N - 1] - x[N] + F
    # then the general case
    for n in 3:(N - 1)
      dx[n] = (x[n + 1] - x[n - 2]) * x[n - 1] - x[n] + F
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
    @inbounds function duffing_eom(dx, x, p, t)
        ω, f, d, β = p
        dx[1] = x[2]
        dx[2] = f*cos(ω*t) - β*x[1] - x[1]^3 - d * x[2]
        return nothing
    end
    @inbounds function  duffing_jacob(J, u, p, t)
        ω, f, d, β = p
        J[2,1] = -β - 3u[1]^2
        J[2,2] = -d
    end
    return ContinuousDS(u0, duffing_eom, duffing_jacob, J; parameters = [ω, f, d, β])
end


"""
    shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)
Shinriki oscillator with all other parameters (besides `R1`) set to constants.
*This is a stiff problem, be careful when choosing solvers and tolerances*.
"""
function shinriki(u0 = [-2, 0, 0.2]; R1 = 22.0)

    shinriki_voltage(V) = 2.295e-5*(exp(3.0038*V) - exp(-3.0038*V))
    function shinriki_eom(du, u, p, t)
        R1 = p[1]

        du[1] = (1/0.01)*(
        u[1]*(1/6.9 - 1/R1) - shinriki_voltage(u[1] - u[2]) - (u[1] - u[2])/14.5
        )

        du[2] = (1/0.1)*(
        shinriki_voltage(u[1] - u[2]) + (u[1] - u[2])/14.5 - u[3]
        )

        du[3] = (1/0.32)*(-u[3]*0.1 + u[2])
        return nothing
    end

    # # Jacobian caller for Shinriki:
    # shinriki_eom(::Type{Val{:jac}}, J, u, p, t) = (shi::Shinriki)(t, u, J)

    return ContinuousDS(u0, shinriki_eom; parameters = [R1])
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

    function gissinger_eom(du, u, p, t)
        μ, ν, Γ = p
        du[1] = μ*u[1] - u[2]*u[3]
        du[2] = -ν*u[2] + u[1]*u[3]
        du[3] = Γ - u[3] + u[1]*u[2]
        return nothing
    end

    return ContinuousDS(u0, gissinger_eom; parameters = [μ, ν, Γ])
end

#######################################################################################
#                                     Discrete                                        #
#######################################################################################
"""
```julia
towel(u0 = [0.085, -0.121, 0.075])
```
```math
\\begin{aligned}
x_{n+1} &= a x_n (1-x_n) -0.05 (y_n +0.35) (1-2z_n) \\\\
y_{n+1} &= 0.1 \\left( \\left( y_n +0.35 \\right)\\left( 1+2z_n\\right) -1 \\right)
\\left( 1 -1.9 x_n \\right) \\\\
z_{n+1} &= 3.78 z_n (1-z_n) + b y_n
\\end{aligned}
```
The folded-towel map is a hyperchaotic mapping due to Rössler [1]. It is famous
for being a mapping that has the smallest possible dimensions necessary for hyperchaos,
having two positive and one negative Lyapunov exponent.
The name comes from the fact that when plotted looks like a folded towel, in every
projection.

Default values are the ones used in the original paper.

[1] : O. E. Rössler, Phys. Lett. **71A**, pp 155 (1979)
"""
function towel(u0=[0.085, -0.121, 0.075])
    @inline @inbounds function eom_towel(x, p)
    x1, x2, x3 = x[1], x[2], x[3]
    SVector( 3.8*x1*(1-x1) - 0.05*(x2+0.35)*(1-2*x3),
    0.1*( (x2+0.35)*(1-2*x3) - 1 )*(1 - 1.9*x1),
    3.78*x3*(1-x3)+0.2*x2 )
    end

    @inline @inbounds function jacob_towel(x, p)
        @SMatrix [3.8*(1 - 2x[1]) -0.05*(1-2x[3]) 0.1*(x[2] + 0.35);
        -0.19((x[2] + 0.35)*(1-2x[3]) - 1)  0.1*(1-2x[3])*(1-1.9x[1])  -0.2*(x[2] + 0.35)*(1-1.9x[1]);
        0.0  0.2  3.78(1-2x[3]) ]
    end
    return DiscreteDS(u0, eom_towel, jacob_towel)
end# should result in lyapunovs: [0.432207,0.378834,-3.74638]

"""
```julia
standardmap(u0=0.001rand(2); k = 0.971635)
```
```math
\\begin{aligned}
\\theta_{n+1} &= \\theta_n + p_{n+1} \\\\
p_{n+1} &= p_n + k\\sin(\\theta_n)
\\end{aligned}
```
The standard map (also known as Chirikov standard map) is a two dimensional,
area-preserving chaotic mapping due to Chirikov [1]. It is one of the most studied
chaotic systems and by far the most studied Hamiltonian (area-preserving) mapping.

The map corresponds to the  Poincaré's surface of section of the kicked rotor system.
Changing the non-linearity parameter `k` transitions the system from completely periodic
motion, to quasi-periodic, to local chaos (mixed phase-space) and finally to global
chaos.

The default parameter `k` is the critical parameter where the golden-ratio torus is
destroyed, as was calculated by Greene [2]. The e.o.m. considers the angle variable
`θ` to be the first, and the angular momentum `p` to be the second, while
both variables
are always taken modulo 2π (the mapping is on the [0,2π)² torus).

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : B. V. Chirikov, Preprint N. **267**, Institute of
Nuclear Physics, Novosibirsk (1969)

[2] : J. M. Greene, J. Math. Phys. **20**, pp 1183 (1979)
"""
function standardmap(u0=0.001rand(2); k = 0.971635)
    @inline @inbounds function standardmap_eom(x, par)
        theta = x[1]; p = x[2]
        p += par[1]*sin(theta)
        theta += p
        while theta >= twopi; theta -= twopi; end
        while theta < 0; theta += twopi; end
        while p >= twopi; p -= twopi; end
        while p < 0; p += twopi; end
        return SVector(theta, p)
    end
    @inline @inbounds standardmap_jacob(x, p) =
    @SMatrix [1 + p[1]*cos(x[1])    1;
              p[1]*cos(x[1])        1]
    return DiscreteDS(u0, standardmap_eom, standardmap_jacob; parameters = [k])
end

"""
```julia
coupledstandardmaps(M::Int, u0 = 0.001rand(2M); ks = ones(M), Γ = 1.0)
```
```math
\\begin{aligned}
\\theta_{i}' &= \\theta_i + p_{i}' \\\\
p_{i}' &= p_i + k_i\\sin(\\theta_i) - \\Gamma \\left[
\\sin(\\theta_{i+1} - \\theta_{i}) + \\sin(\\theta_{i-1} - \\theta_{i})
\\right]
\\end{aligned}
```
A discrete system of `M` nonlinearly coupled standard maps, first
introduced in [1] to study diffusion and chaos thresholds.
The *total* dimension of the system
is `2M`. The maps are coupled through `Γ`
and the `i`-th map has a nonlinear parameter `ks[i]`.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : H. Kantz & P. Grassberger, J. Phys. A **21**, pp 127–133 (1988)
"""
function coupledstandardmaps(M::Int, u0 = 0.001rand(2M);
    ks = ones(M), Γ = 1.0)

    idxs = 1:M # indexes of thetas
    idxsm1 = circshift(idxs, +1)  #indexes of thetas - 1
    idxsp1 = circshift(idxs, -1)  #indexes of thetas + 1

    csm = CoupledStandardMaps{M}(idxs, idxsm1, idxsp1)
    J = zeros(eltype(u0), 2M, 2M)
    # Set ∂/∂p entries (they are eye(M,M))
    # And they dont change they are constants
    for i in idxs
        J[i, i+M] = 1
        J[i+M, i+M] = 1
    end

    return BigDiscreteDS(u0, csm, csm, J; parameters = [ks, Γ])
end
mutable struct CoupledStandardMaps{N}
    idxs::UnitRange{Int}
    idxsm1::Vector{Int}
    idxsp1::Vector{Int}
end
@inbounds function (f::CoupledStandardMaps{N})(xnew::AbstractVector, x, p) where {N}
    ks, Γ = p
    for i in f.idxs

        xnew[i+N] = mod2pi(
            x[i+N] + ks[i]*sin(x[i]) -
            Γ*(sin(x[f.idxsp1[i]] - x[i]) + sin(x[f.idxsm1[i]] - x[i]))
        )

        xnew[i] = mod2pi(x[i] + xnew[i+N])
    end
    return nothing
end
@inbounds function (f::CoupledStandardMaps{M})(
    J::AbstractMatrix, x, p) where {M}

    ks, Γ = p
    # x[i] ≡ θᵢ
    # x[[idxsp1[i]]] ≡ θᵢ+₁
    # x[[idxsm1[i]]] ≡ θᵢ-₁
    for i in f.idxs
        cosθ = cos(x[i])
        cosθp= cos(x[f.idxsp1[i]] - x[i])
        cosθm= cos(x[f.idxsm1[i]] - x[i])
        J[i+M, i] = ks[i]*cosθ + Γ*(cosθp + cosθm)
        J[i+M, f.idxsm1[i]] = - Γ*cosθm
        J[i+M, f.idxsp1[i]] = - Γ*cosθp
        J[i, i] = 1 + J[i+M, i]
        J[i, f.idxsm1[i]] = J[i+M, f.idxsm1[i]]
        J[i, f.idxsp1[i]] = J[i+M, f.idxsp1[i]]
    end
    return nothing
end


"""
```julia
henon(u0=zeros(2); a = 1.4, b = 0.3)
```
```math
\\begin{aligned}
x_{n+1} &= 1 - ax^2_n+y_n \\\\
y_{n+1} & = bx_n
\\end{aligned}
```
The Hénon map is a two-dimensional mapping due to Hénon [1] that can display a strange
attractor (at the default parameters). In addition, it also displays many other aspects
of chaos, like period doubling or intermittency, for other parameters.

According to the author, it is a system displaying all the properties of the
Lorentz system (1963) while being as simple as possible.
Default values are the ones used in the original paper.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : M. Hénon, Commun.Math. Phys. **50**, pp 69 (1976)
"""
function henon(u0=zeros(2); a = 1.4, b = 0.3)
    henon_eom(x, p) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
    henon_jacob(x, p) = @SMatrix [-2*p[1]*x[1] 1.0; p[2] 0.0]
    return DiscreteDS(u0, henon_eom, henon_jacob; parameters = [a, b])
end # should give lyapunov exponents [0.4189, -1.6229]


"""
```julia
logistic(x0 = rand(); r = 4.0)
```
```math
x_{n+1} = rx_n(1-x_n)
```
The logistic map is an one dimensional unimodal mapping due to May [1] and is used by
many as the archetypal example of how chaos can arise from very simple equations.

Originally intentend to be a discretized model of polulation dynamics, it is now famous
for its bifurcation diagram, an immensly complex graph that that was shown
be universal by Feigenbaum [2].

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : R. M. May, Nature **261**, pp 459 (1976)

[2] : M. J. Feigenbaum, J. Stat. Phys. **19**, pp 25 (1978)
"""
function logistic(x0=rand(); r = 4.0)
    @inline logistic_eom(x, p) = p[1]*x*(1-x)
    @inline logistic_jacob(x, p) = p[1]*(1-2x)
    return DiscreteDS1D(x0, logistic_eom, logistic_jacob; parameters = [r])
end



"""
    circlemap(x0=rand(); Ω = 1.0, K = 0.99)

```math
\\theta_{n+1} = \\theta_n + 2\\pi\\Omega - K\\sin(\\theta_n)
```
Notice that the map *does not use* `mod2pi` at the equation of motion.

The parameter container has the parameters in the same order as stated in this
function's documentation string.
"""
function circlemap(x0=rand(); K = 0.99, Ω = 1.0, usemod::Bool = false)
    @inline circlemap_eom(x, p) = x + twopi*p[1] - p[2]*sin(x)
    @inline circlemap_jacob(x, p) = -p[2]*cos(x)
    return DiscreteDS1D(x0, circlemap_eom, circlemap_jacob; parameters = [Ω, K])
end



end# Systems module
