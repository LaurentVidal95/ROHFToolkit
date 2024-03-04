import Base: getindex, setindex!, push!, pop!, popfirst!, empty!

abstract type Solver end

@doc raw"""
    OLD: GradientDescentManual(; preconditioned=true)

(Preconditioned) Steepest descent algorithm on the AMO manifold.
"""
struct GradientDescentManual <: Solver
    name           ::String
    prefix         ::String
end
function GradientDescentManual(; preconditioned=true)
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
    prefix = preconditioned ? "prec_SD" : "SD"
    @info name
    GradientDescentManual(name, prefix)
end

function next_dir(S::GradientDescentManual, info; preconditioner, kwargs...)
    grad_kappa = preconditioner(info.∇E)
    dir = TangentVector(-grad_kappa, info.ζ)
    dir, merge(info, (; dir))
end


@doc raw"""
    OLD: ConjugateGradientManual(; preconditioned=true, flavor="Fletcher-Reeves")

(Preconditioned) conjugate gradient algorithm on the MO manifold.
The ``cg_type`` for now is useless but will serve to launch other
types of CG algorithms.
"""
struct ConjugateGradientManual <: Solver
    name           ::String
    prefix         ::String
    flavor         ::Symbol
end
function ConjugateGradientManual(; preconditioned=true, flavor=:Fletcher_Reeves)
    @assert flavor ∈ (:Fletcher_Reeves, :Polack_Ribiere, :Hestenes_Stiefel)
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    @info "$(name) with flavor: $(flavor)"
    ConjugateGradientManual(name, prefix, flavor)
end

function next_dir(S::ConjugateGradientManual, info; preconditioner, transport_type)
    x_new = info.ζ
    ∇E = info.∇E; ∇E_prev = info.∇E_prev
    P∇E = info.P∇E; P∇E_prev = info.P∇E_prev
    dir=info.dir; x_prev = dir.base

    # Transport previous dir and gradient on current point x_new
    τdir = transport_AMO(dir, x_prev, dir, info.step, x_new;
                         type=transport_type, collinear=true)

    function cg_coeff(flavor)
        (flavor==:Polack_Ribiere) && (return tr(∇E'P∇E)/tr(∇E_prev'P∇E_prev))
        τ_P∇E_prev = transport_AMO(P∇E_prev, x_prev, dir, info.step, x_new;
                                   type=transport_type, collinear=false)
        (flavor==:Fletcher_Reeves) && (return (tr(∇E'P∇E) - tr(∇E'τ_P∇E_prev)) / tr(∇E_prev'P∇E_prev))
        # Hestenes Stiefel (to be confirmed)
        return (tr(∇E'P∇E) - tr(∇E'τ_P∇E_prev)) / (tr(τdir'P∇E) - tr(dir'P∇E_prev))
    end
    
    β = cg_coeff(S.flavor)
    dir = TangentVector(-P∇E + β*τdir, x_new)

    # Restart if new dir is not a descent direction
    (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2) && (β = zero(β))
    iszero(β) && (@warn "Restart"; dir = TangentVector(-∇E, x_new))

    dir, merge(info, (; dir))
end

"""
LBFGS on the AMO manifold.
"""
struct LBFGSManual <: Solver
    name           ::String
    prefix         ::String
    depth          ::Int
    B₀             ::Function
end
function LBFGSManual(;depth=8, B₀=default_LBFGS_init, preconditioned=true)
    name = preconditioned ? "Preconditioned LBFGS" : "LBFGS"
    prefix = preconditioned ? "prec_LBFGS" : "LBFGS"
    @info "$(name) with depth $(depth)"
    LBFGSManual(name, prefix, depth, B₀)
end

"""
For now, no preconditioning.
Not supposed to work with other transports and rectractions than exp.
"""
function next_dir(S::LBFGSManual, info; preconditioner, transport_type=:exp)
    # Extract data
    B = info.B
    x_prev = info.dir.base;  ∇E_prev = info.∇E_prev
    x_new = info.ζ; ∇E = info.∇E
    P∇E = info.P∇E; P∇E_prev = info.P∇E_prev;
    dir = info.dir
    # renaming for small lines
    type=transport_type
    
    # Transport previous s and y to current location
    if B.length ≥ 1
        for k in 1:B.length
            s, y, Py, ρ = B[k]
            s = transport_AMO(s, x_prev, dir, info.step, x_new; type, collinear=false)
            y = transport_AMO(y, x_prev, dir, info.step, x_new; type, collinear=false)
            Py = transport_AMO(Py, x_prev, dir, info.step, x_new; type, collinear=false)

            # Project back on the tangent plane if s or y propagate errors
            # s = TangentVector(project_tangent_AMO(x_new, s.vec), x_new)
            # y = TangentVector(project_tangent_AMO(x_new, y.vec), x_new)

            s_tangent = is_tangent(s; tol=1e-9, return_value=true)
            y_tangent = is_tangent(y; tol=1e-9, return_value=true)
            Py_tangent = is_tangent(Py; tol=1e-9, return_value=true)
            if !(s_tangent[1] && y_tangent[1] && Py_tangent[1])
                @warn "Direction out of the tangent plane: \n"*
                    "test s: $(s_tangent[2])\n test_y: $(y_tangent[2])\n test_Py: $(Py_tangent[2])"
            end
            B[k] = (s,y,Py,ρ)
        end
    end

    # Compute current s, y and ρ.
    αdir = TangentVector(info.step*dir.kappa, dir.base)
    s = transport_AMO(αdir, x_prev, dir, info.step, x_new; type, collinear=true)
    y = TangentVector(∇E.kappa - transport_AMO(∇E_prev, x_prev, dir, info.step, x_new;
                                             type, collinear=false).kappa,
                      x_new)

    Py = TangentVector(P∇E.kappa - transport_AMO(P∇E_prev, x_prev, dir, info.step, x_new;
                                                 type, collinear=false).kappa, x_new)
    
    ρ = 1/tr(s'y)
    push!(B, (s,y,Py,ρ))

    # Compute next dir
    dir_kappa = -B(∇E; preconditioner, S.B₀)
    dir = TangentVector(dir_kappa, x_new)

    # Restart BFGS if dir is not a descent direction
    if (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2)
        @warn "Restart: not a descent direction"
        empty!(B)
        dir = TangentVector(-preconditioner(∇E), info.ζ)
    end
    # Restart if the step is too small
    if (info.step < 1e-4)
        @warn "Restart: steps is too small"
        empty!(B)
        dir = TangentVector(-preconditioner(∇E), info.ζ)
    end
    # if info.n_iter==130
    #     @warn "Forced"
    #     empty!(B)
    #     dir = TangentVector(-preconditioner(∇E), info.ζ)
    # end

    # DEBUG : norm(dir) goes to zero every 30 iterations... Why ?
    dir, merge(info, (; dir, B))
end


mutable struct LBFGSInverseHessian{TangentType,ScalarType}
    maxlength::Int
    length::Int
    first::Int
    S ::Vector{TangentType}
    Y ::Vector{TangentType}
    PY::Vector{TangentType} # preconditioned version of Y vector
    ρ ::Vector{ScalarType}
    α ::Vector{ScalarType} # work space
    function LBFGSInverseHessian{T1,T2}(maxlength::Int, S::Vector{T1}, Y::Vector{T1},
                                        PY::Vector{T1}, ρ::Vector{T2}) where {T1, T2}
        @assert length(S) == length(Y) == length(ρ) == length(PY)
        l = length(S)
        S = resize!(copy(S), maxlength)
        Y = resize!(copy(Y), maxlength)
        PY = resize!(copy(PY), maxlength)
        ρ = resize!(copy(ρ), maxlength)
        α = similar(ρ)
        return new{T1,T2}(maxlength, l, 1, S, Y, PY, ρ, α)
    end
end
LBFGSInverseHessian(maxlength::Int, S::Vector{T1}, Y::Vector{T1}, PY::Vector{T1},
                    ρ::Vector{T2}) where {T1, T2} = LBFGSInverseHessian{T1, T2}(maxlength, S, Y, PY, ρ)

function Absil_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector)
    s, y, Py, ρ = B[B.length]
    γ = tr(s'y)/tr(y'Py)
    Nb = size(g.kappa,1)
    γ*Matrix(I, Nb, Nb)
end

default_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector) = Absil_LBFGS_init(B, g)
id_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector) = I

function EWC_LBFGS_guess(B::LBFGSInverseHessian, g::TangentVector)
    error("TO DEBUG")
    # x = g.base
    # Nb, Ni, Na = x.Σ.mo_numbers
    # G = ROHF_ambiant_gradient(x)
    # # Only keep the diagonal blocs of the gradient
    # G_diag = zero(G)
    # G_diag[1:Ni, 1:Ni] .= G[1:Ni, 1:Ni]
    # G_diag[Ni+1:Ni+Na, Ni+1:Ni+Na] .= G[Ni+1:Ni+Na, Ni+1:Ni+Na]
    # G_diag[Ni+Na+1:Nb, Ni+Na+1:Nb] .= G[Ni+Na+1:Nb, Ni+Na+1:Nb]
    # # Diagonalize and assemble pseudo hessian
    # λs = eigvals(G_diag)
    # diagm(λs)
end
"""
Compute next dir using the two-loop L-BFGS evalutation
"""
function (B::LBFGSInverseHessian)(g::TangentVector; preconditioner, B₀=default_LBFGS_init)
    q = deepcopy(g)
    α = zeros(eltype(g.base.Φ), B.length)
    # First loop
    for i = B.length:-1:1
        s, y, Py, ρ = B[i]
        α[i] = ρ * tr(s'q)
        q = TangentVector(q.kappa - α[i]*y, q.base)
    end
    # Compute Bₖ⁰q
    Pq = preconditioner(q)
    r = TangentVector(B₀(B, g)*Pq.kappa, q.base)
    # Second loop
    for i = 1:B.length
        s, y, Py, ρ = B[i]
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
    return (getindex(B.S, idx), getindex(B.Y, idx), getindex(B.PY, idx), getindex(B.ρ, idx))
end

@inline function Base.setindex!(B::LBFGSInverseHessian, (s, y, Py, ρ), i)
    @boundscheck if i < 1 || i > B.length
        throw(BoundsError(B, i))
    end
    n = B.maxlength
    idx = B.first + i - 1
    idx = ifelse(idx > n, idx - n, idx)
    (setindex!(B.S, s, idx), setindex!(B.Y, y, idx), setindex!(B.PY, y, idx), setindex!(B.ρ, ρ, idx))
end

@inline function Base.push!(B::LBFGSInverseHessian, (s, y, Py, ρ))
    if B.length < B.maxlength
        B.length += 1
    else
        B.first = (B.first == B.maxlength ? 1 : B.first + 1)
    end
    @inbounds setindex!(B, (s, y, Py, ρ), B.length)
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
