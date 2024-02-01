import Base: getindex, setindex!, push!, pop!, popfirst!, empty!

abstract type Solver end

@doc raw"""
    OLD: GradientDescentManual(; preconditioned=true)

(Preconditioned) Steepest descent algorithm on the AMO manifold.
"""
struct GradientDescentManual <: Solver
    name           ::String
    prefix         ::String
    preconditioned ::Bool
    linesearch
end
function GradientDescentManual(; preconditioned=true, linesearch)
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
    prefix = preconditioned ? "prec_SD" : "SD"
    GradientDescentManual(name, prefix, preconditioned, linesearch)
end

function next_dir(S::GradientDescentManual, info)
    grad_vec = S.preconditioned ? preconditioned_gradient_AMO(info.ζ) : info.∇E
    dir = TangentVector(-grad_vec, info.ζ)
    dir, merge(info, (; dir))
end


@doc raw"""
    OLD: ConjugateGradientManual(; preconditioned=true, flavour="Fletcher-Reeves")

(Preconditioned) conjugate gradient algorithm on the MO manifold.
The ``cg_type`` for now is useless but will serve to launch other
types of CG algorithms.
"""
struct ConjugateGradientManual <: Solver
    name           ::String
    prefix         ::String
    preconditioned ::Bool
    flavor        ::Symbol
    transport      ::Symbol
    linesearch
end
function ConjugateGradientManual(; preconditioned=true, flavor=:Fletcher_Reeves,
                                 transport_type=:exp, linesearch)
    @assert flavor ∈ (:Fletcher_Reeves, :Polack_Ribiere)
    @assert transport_type ∈ (:exp, :QR, :proj)
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    ConjugateGradientManual(name, prefix, preconditioned, flavor, transport_type, linesearch)
end


function next_dir(S::ConjugateGradientManual, info)
    ζ = info.ζ
    ∇E = info.∇E;  ∇E_prev = info.∇E_prev;
    dir = info.dir
    current_grad = S.preconditioned ? preconditioned_gradient_AMO(ζ) : ∇E

    # Transport previous dir and gradient on current point ζ
    τ_dir_prev = transport_AMO(dir, dir.base, dir, 1., ζ; type=S.transport, collinear=true)

    β = zero(ζ.energy)
    begin
        cg_factor = zero(β)
        if S.flavor==:Fletcher_Reeves
            τ_grad_prev = transport_AMO(∇E_prev, ∇E_prev.base, dir, 1., ζ; type=S.transport, collinear=false)
            cg_factor = tr(∇E'τ_grad_prev)
        end
        β = (tr(∇E'∇E) - cg_factor) / norm(info.∇E_prev)^2 # DEBUG: wrong use of preconditioning ?
        # Restart if not a descent direction
        dir = TangentVector(project_tangent_AMO(ζ, -current_grad + β*τ_dir_prev), ζ)
        (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2) && (β = zero(β))
        # Restart if β is negative
        β = (β > 0) ? β : zero(Float64)
    end
    iszero(β) && (@warn "Restart"; dir = TangentVector(-current_grad, ζ))

    dir, merge(info, (; dir))
end


"""
LBFGS on the AMO manifold.
"""
struct LBFGSManual <: Solver
    name           ::String
    prefix         ::String
    preconditioned ::Bool
    depth          ::Int
    B₀             ::Function
    linesearch
end
function LBFGSManual(;depth=8, B₀=default_LBFGS_init, preconditioned=:false, linesearch)
    name = preconditioned ? "Preconditioned LBFGS" : "LBFGS"
    prefix = preconditioned ? "prec_LBFGS" : "LBFGS"
    LBFGSManual(name, prefix, preconditioned, depth, B₀, linesearch)
end

function next_dir(S::LBFGSManual, info)
    # Extract data
    B = info.B
    x_prev = info.dir.base;  ∇E_prev = info.∇E_prev
    x_new = info.ζ; ∇E = info.∇E
    dir = info.dir

    # Transport previous s and y to current location
    if B.length ≥ 1
        for k in 1:B.length
            s, y, ρ = B[k]
            s = transport_AMO(s, x_prev, dir, 1., x_new; type=:exp, collinear=false)
            y = transport_AMO(y, x_prev, dir, 1., x_new; type=:exp, collinear=false)

            # Project back on the tangent plane if s or y propagate errors
            s = TangentVector(project_tangent_AMO(x_new, s.vec), x_new)
            y = TangentVector(project_tangent_AMO(x_new, y.vec), x_new)

            sy_tangents = all([test_tangent(s)<1e-9, test_tangent(y)<1e-9])
            !(sy_tangents) && (@show sy_tangents)
            B[k] = (s,y,ρ)
        end
    end

    # Compute current s, y and ρ.
    s = transport_AMO(dir, x_prev, dir, 1., x_new; type=:exp, collinear=true)
    y = TangentVector(∇E.vec - transport_AMO(∇E_prev, x_prev, dir, 1., x_new;
                                             type=:exp, collinear=false
                                             ).vec,
                      x_new)
    ρ = 1/tr(s'y)
    push!(B, (s,y,ρ))

    # Compute next dir
    dir_vec = -B(∇E; S.B₀)
    dir = TangentVector(-B(∇E).vec, x_new)
    
    # Restart BFGS if dir is not a descent direction
    if (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2)
        @warn "Restart: not a descent direction"
        empty!(B)
        dir = TangentVector(-∇E, info.ζ)
    end
    # DEBUG : norm(dir) goes to zero every 30 iterations... Why ?

    dir, merge(info, (; dir, B))
end


mutable struct LBFGSInverseHessian{TangentType,ScalarType}
    maxlength::Int
    length::Int
    first::Int
    S::Vector{TangentType}
    Y::Vector{TangentType}
    ρ::Vector{ScalarType}
    α::Vector{ScalarType} # work space
    function LBFGSInverseHessian{T1,T2}(maxlength::Int, S::Vector{T1}, Y::Vector{T1},
                                        ρ::Vector{T2}) where {T1, T2}
        @assert length(S) == length(Y) == length(ρ)
        l = length(S)
        S = resize!(copy(S), maxlength)
        Y = resize!(copy(Y), maxlength)
        ρ = resize!(copy(ρ), maxlength)
        α = similar(ρ)
        return new{T1,T2}(maxlength, l, 1, S, Y, ρ, α)
    end
end
LBFGSInverseHessian(maxlength::Int, S::Vector{T1}, Y::Vector{T1},
                    ρ::Vector{T2}) where {T1, T2} = LBFGSInverseHessian{T1, T2}(maxlength, S, Y, ρ)

function Absil_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector)
    @show "ABSIL"
    s, y, ρ = B[B.length]
    γ = tr(s'y)/(norm(y)^2)
    Nb = size(g.vec,1)
    γ*Matrix(I, Nb, Nb)
end

default_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector) = I

"""
Compute next dir using the two-loop L-BFGS evalutation
"""
function (B::LBFGSInverseHessian)(g::TangentVector; B₀=default_LBFGS_init)
    q = deepcopy(g)
    α = zeros(eltype(g.base.Φ), B.length)
    # First loop
    for i = B.length:-1:1
        s, y, ρ = B[i]
        α[i] = ρ * tr(s'q)
        q = TangentVector(q.vec - α[i]*y, q.base)
    end
    # Compute Bₖ⁰q
    r = TangentVector(B₀(B, g)*q.vec, q.base)
    # Second loop
    for i = 1:B.length
        s, y, ρ = B[i]
        β = ρ * tr(y'r)
        r = TangentVector(r + (α[i]-β)*s, r.base)
    end
    # Return new direction
    return r
end

@inline function Base.getindex(B::LBFGSInverseHessian, i::Int)
    @boundscheck if i < 1 || i > B.length
        throw(BoundsError(B, i))
    end
    n = B.maxlength
    idx = B.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    return (getindex(B.S, idx), getindex(B.Y, idx), getindex(B.ρ, idx))
end

@inline function Base.setindex!(B::LBFGSInverseHessian, (s, y, ρ), i)
    @boundscheck if i < 1 || i > B.length
        throw(BoundsError(B, i))
    end
    n = B.maxlength
    idx = B.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    (setindex!(B.S, s, idx), setindex!(B.Y, y, idx), setindex!(B.ρ, ρ, idx))
end

@inline function Base.push!(B::LBFGSInverseHessian, (s, y, ρ))
    if B.length < B.maxlength
        B.length += 1
    else
        B.first = (B.first == B.maxlength ? 1 : B.first + 1)
    end
    @inbounds setindex!(B, (s, y, ρ), B.length)
    return B
end
@inline function Base.pop!(B::LBFGSInverseHessian)
    @inbounds v = B[B.length]
    B.length -= 1
    return v
end
@inline function Base.popfirst!(B::LBFGSInverseHessian)
    @inbounds v = B[1]
    B.first = (B.first == B.maxlength ? 1 : B.first + 1)
    B.length -= 1
    return v
end

@inline function Base.empty!(B::LBFGSInverseHessian)
    B.length = 0
    B.first = 1
    return B
end
