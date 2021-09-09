"""
```julia
towel(u0 = [0.085, -0.121, 0.075])
```
```math
\\begin{aligned}
x_{n+1} &= 3.8 x_n (1-x_n) -0.05 (y_n +0.35) (1-2z_n) \\\\
y_{n+1} &= 0.1 \\left[ \\left( y_n +0.35 \\right)\\left( 1+2z_n\\right) -1 \\right]
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
    return DDS(rule_towel, u0, nothing, jacob_towel)
end# should result in lyapunovs: [0.432207,0.378834,-3.74638]
function rule_towel(x, p, n)
    @inbounds x1, x2, x3 = x[1], x[2], x[3]
    SVector( 3.8*x1*(1-x1) - 0.05*(x2+0.35)*(1-2*x3),
    0.1*( (x2+0.35)*(1-2*x3) - 1 )*(1 - 1.9*x1),
    3.78*x3*(1-x3)+0.2*x2 )
end
function jacob_towel(x, p, n)
    @SMatrix [3.8*(1 - 2x[1]) -0.05*(1-2x[3]) 0.1*(x[2] + 0.35);
    -0.19((x[2] + 0.35)*(1-2x[3]) - 1)  0.1*(1-2x[3])*(1-1.9x[1])  -0.2*(x[2] + 0.35)*(1-1.9x[1]);
    0.0  0.2  3.78(1-2x[3]) ]
end

function rule_towel_iip(dx, x, p, n)
    @inbounds begin
        x1, x2, x3 = x[1], x[2], x[3]
        dx[1] = 3.8*x1*(1-x1) - 0.05*(x2+0.35)*(1-2*x3)
        dx[2] = 0.1*( (x2+0.35)*(1-2*x3) - 1 )*(1 - 1.9*x1)
        dx[3] = 3.78*x3*(1-x3)+0.2*x2
    end
end
function jacob_towel_iip(J, x, p, n)
    @inbounds begin
        J[1,1] = 3.8*(1 - 2x[1])
        J[2,1] = -0.19((x[2] + 0.35)*(1-2x[3]) - 1)
        J[3,1] = 0.0
        J[1,2] = -0.05*(1-2x[3])
        J[2,2] =  0.1*(1-2x[3])*(1-1.9x[1])
        J[3,2] = 0.2
        J[1,3] = 0.1*(x[2] + 0.35)
        J[2,3] = -0.2*(x[2] + 0.35)*(1-1.9x[1])
        J[3,3] = 3.78(1-2x[3])
    end
end




"""
```julia
standardmap(u0=[0.001245, 0.00875]; k = 0.971635)
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
function standardmap(u0=[0.001245, 0.00875]; k = 0.971635)
    return DDS(standardmap_rule, u0, [k], standardmap_jacob)
end
@inbounds function standardmap_rule(x, par, n)
    theta = x[1]; p = x[2]
    p += par[1]*sin(theta)
    theta += p
    while theta >= twopi; theta -= twopi; end
    while theta < 0; theta += twopi; end
    while p >= twopi; p -= twopi; end
    while p < 0; p += twopi; end
    return SVector(theta, p)
end
@inbounds standardmap_jacob(x, p, n) =
@SMatrix [1 + p[1]*cos(x[1])    1;
          p[1]*cos(x[1])        1]

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
The first `M` parameters are the `ks`, the `M+1`th parameter is `Γ`.

The first `M` entries of the state are the angles, the last `M` are the momenta.

[1] : H. Kantz & P. Grassberger, J. Phys. A **21**, pp 127–133 (1988)
"""
function coupledstandardmaps end
using SparseArrays
function coupledstandardmaps(M::Int, u0 = 0.001rand(2M);
    ks = ones(M), Γ = 1.0)

    SV = SVector{M, Int}
    idxs = SV(1:M...) # indexes of thetas
    idxsm1 = SV(circshift(idxs, +1)...)  #indexes of thetas - 1
    idxsp1 = SV(circshift(idxs, -1)...)  #indexes of thetas + 1

    csm = CoupledStandardMaps{M}(idxs, idxsm1, idxsp1)
    J = zeros(eltype(u0), 2M, 2M)
    # Set ∂/∂p entries (they are eye(M,M))
    # And they dont change they are constants
    for i in idxs
        J[i, i+M] = 1
        J[i+M, i+M] = 1
    end
    sparseJ = sparse(J)
    p = vcat(ks, Γ)
    csm(sparseJ, u0, p, 0)
    return DDS(csm, u0, p, csm, sparseJ)
end
struct CoupledStandardMaps{N}
    idxs::SVector{N, Int}
    idxsm1::SVector{N, Int}
    idxsp1::SVector{N, Int}
end
function (f::CoupledStandardMaps{N})(xnew::AbstractVector, x, p, n) where {N}
    ks = view(p, 1:N); Γ = p[end]
    @inbounds for i in f.idxs

        xnew[i+N] = mod2pi(
            x[i+N] + ks[i]*sin(x[i]) -
            Γ*(sin(x[f.idxsp1[i]] - x[i]) + sin(x[f.idxsm1[i]] - x[i]))
        )

        xnew[i] = mod2pi(x[i] + xnew[i+N])
    end
    return xnew
end
function (f::CoupledStandardMaps{M})(
    J::AbstractMatrix, x, p, n) where {M}

    ks = view(p, 1:M); Γ = p[end]
    # x[i] ≡ θᵢ
    # x[[idxsp1[i]]] ≡ θᵢ+₁
    # x[[idxsm1[i]]] ≡ θᵢ-₁
    @inbounds for i in f.idxs
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
    return J
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
    return DDS(hoop, u0, [a,b], hoop_jac)
end # should give lyapunov exponents [0.4189, -1.6229]
hoop(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])
hoop_jac(x, p, n) = @SMatrix [-2*p[1]*x[1] 1.0; p[2] 0.0]

function henon_iip(u0=zeros(2); a = 1.4, b = 0.3)
    return DDS(hiip, u0, [a, b], hiip_jac)
end
function hiip(dx, x, p, n)
    dx[1] = 1.0 - p[1]*x[1]^2 + x[2]
    dx[2] = p[2]*x[1]
    return
end
function hiip_jac(J, x, p, n)
    J[1,1] = -2*p[1]*x[1]
    J[1,2] = 1.0
    J[2,1] = p[2]
    J[2,2] = 0.0
    return
end


"""
```julia
logistic(x0 = 0.4; r = 4.0)
```
```math
x_{n+1} = rx_n(1-x_n)
```
The logistic map is an one dimensional unimodal mapping due to May [1] and is used by
many as the archetypal example of how chaos can arise from very simple equations.

Originally intentend to be a discretized model of polulation dynamics, it is now famous
for its bifurcation diagram, an immensely complex graph that that was shown
be universal by Feigenbaum [2].

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[1] : R. M. May, Nature **261**, pp 459 (1976)

[2] : M. J. Feigenbaum, J. Stat. Phys. **19**, pp 25 (1978)
"""
function logistic(x0=0.4; r = 4.0)
    return DDS(logistic_rule, x0, [r], logistic_jacob)
end
logistic_rule(x, p, n) = p[1]*x*(1-x)
logistic_jacob(x, p, n) = p[1]*(1-2x)

"""
    pomaeu_manneville(u0 = 0.2; z = 2.5)
The Pomeau-Manneville map is a one dimensional discrete map which is
characteristic for displaying intermittency [1]. Specifically, for
z > 2 the average time between chaotic bursts diverges, while
for z > 2.5, the map iterates are long range correlated [2].

Notice that here we are providing the "symmetric" version:
```math
x_{n+1} = \\begin{cases}
-4x_n + 3, & \\quad x_n \\in (0.5, 1] \\\\
x_n(1 + |2x_n|^{z-1}), & \\quad |x_n| \\le 0.5 \\\\
-4x_n - 3, & \\quad x_n \\in [-1, 0.5)
\\end{cases}
```

[1] : Manneville & Pomeau, Comm. Math. Phys. **74** (1980)

[2] : Meyer et al., New. J. Phys **20** (2019)
"""
function pomeau_manneville(u0 = 0.2, z = 2.5)
    return DDS(pm_rule, u0, [z], pm_jac)
end
function pm_rule(x, p, n)
    if x < -0.5
        -4x - 3
    elseif -0.5 ≤ x ≤ 0.5
        @inbounds x*(1 + abs(2x)^(p[1]-1))
    else
        -4x + 3
    end
end
function pm_jac(x, p, n)
    if x < -0.5
        -4.0
    elseif -0.5 ≤ x ≤ 0.5
        @inbounds z = p[1]
        0.5(x^2 * 2^z * (z-1)*abs(x)^(z-3) + 2^z * abs(x)^(z-1) + 2)
    else
        -4.0
    end
end

"""
```julia
manneville_simple(x0 = 0.4; ε = 1.1)
```
```math
x_{n+1} = [ (1+\\varepsilon)x_n + (1-\\varepsilon)x_n^2 ] \\mod 1
```
A simple 1D map due to Mannevile[^Manneville1980] that is useful in illustrating the concept
and properties of intermittency.

The parameter container has the parameters in the same order as stated in this
function's documentation string.

[^Manneville1980]: Manneville, P. (1980). Intermittency, self-similarity and 1/f spectrum in dissipative dynamical systems. [Journal de Physique, 41(11), 1235–1243](https://doi.org/10.1051/jphys:0198000410110123500)
"""
function manneville_simple(x0=0.4; ε = 0.1)
    return DDS(manneville_f, x0, [ε], manneville_j)
end

function manneville_f(x, p, t)
    e = p[1]
    y = (1+e)*x + (1-e)*x*x
    return y%1
end
manneville_j(x, p, n) = (1+p[1]) + (1-p[1])*2x

"""
```julia
arnoldcat(u0 = [0.001245, 0.00875])
```
```math
f(x,y) = (2x+y,x+y) \\mod 1
```
Arnold's cat map. A chaotic map from the torus into itself, used by
Vladimir Arnold in the 1960s. [1]

[1] : Arnol'd, V. I., & Avez, A. (1968). Ergodic problems of classical mechanics.
"""
function arnoldcat(u0 = [0.001245, 0.00875])
    return DDS(arnoldcat_rule, u0, nothing, arnoldcat_jacob)
end # Should give Lyapunov exponents [2.61803, 0.381966]
function arnoldcat_rule(u, p, n)
    x,y = u
    return SVector{2}((2x + y) % 1.0, (x + y) % 1)
end
arnoldcat_jacob(u, p, n) = @SMatrix [2 1; 1 1]



"""
```julia
grebogi_map(u0 = [0.2, 0.]; a = 1.32, b=0.9, J₀=0.3)
```
```math
\\begin{aligned}
\\theta_{n+1} &= \\theta_n +   a\\sin 2 \\theta_n -b \\sin 4 \\theta_n -x_n\\sin \\theta_n\\\\
x_{n+1} &= -J_0 \\cos \\theta_n
\\end{aligned}
```

This map has two fixed point at `(0,-J_0)` and `(π,J_0)` which are attracting for `|1+2a-4b|<1`. There is a chaotic transient dynamics
before the dynamical systems settles at a fixed point. This map illustrate the fractalization of the basins boundary and its uncertainty exponent `α` is roughly 0.2.

[^Grebogi1983]: C. Grebogi, S. W. McDonald, E. Ott and J. A. Yorke, Final state sensitivity: An obstruction to predictability, Physics Letters A, 99, 9, 1983

## Example
```julia
ds = Systems.grebogi_map(rand(2))
integ = integrator(ds)
θg = range(0, 2π, length=300)
xg = range(-0.5, 0.5, length=300)
bsn, att = basins_map2D(θg, xg, integ)
```
"""
function grebogi_map(u0 = [0.2, 0.]; a = 1.32, b=0.9, J₀=0.3)
    return DDS(grebogi_map_rule, u0, [a,b,J₀], grebogi_map_J)
end
function grebogi_map_rule(u, p, n)
    θ = u[1]; x = u[2]
    a,b,J₀ = p
    dθ= θ + a*sin(2*θ) - b*sin(4*θ) -x*sin(θ)
    dθ = mod(dθ,2π) # to avoid problems with attractor at θ=π
    dx=-J₀*cos(θ)
    return SVector{2}(dθ,dx)
end

function grebogi_map_J(u, p, n)
    θ = u[1]; x = u[2]
    a,b,J₀ = p
    return @SMatrix [(1+2*a*cos(2*θ) - 4*b*cos(4*θ) -x*cos(θ)) J₀*sin(θ); -sin(θ) 0]
end


"""
    nld_coupled_logistic_maps(D = 4, u0 = range(0, 1; length=D); λ = 1.2, k = 0.08)

A high-dimensional discrete dynamical system that couples `D` logistic maps with a
strongly nonlinear all-to-all coupling. 
For the default parameters it displays several co-existing attractors. The equations are:
```math
u_i' = \\lambda - u_i^2 + k \\sum_{j\\ne i} (u_j^2 - u_i^2)
```
Here the prime ``'`` denotes next state.
"""
function nld_coupled_logistic_maps(D = 4, u0 = range(0, 1; length=D); λ = 1.2, k = 0.08)
    return DDS(coupled_logistic_maps_f, u0, [λ, k])
end

function nld_coupled_logistic_maps(du, u, p, n)
    λ, k = p
    D = length(z)
    for i in 1:D
        du[i] = λ - u[i]^2
        for j in 1:D
            j == i && continue
            du[i] += k*(u[j]^2 - u[i]^2)
        end
    end
    return
end
