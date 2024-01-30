import Base: getindex, setindex!, push!, pop!, popfirst!, empty!

@doc raw"""
    OLD: steepest_descent(; preconditioned=true)

(Preconditioned) Steepest descent algorithm on the MO manifold.
"""
function steepest_descent(; preconditioned=true)
    function next_dir(info)
        grad_vec = preconditioned ? preconditioned_gradient_AMO(info.ζ)[1] : info.∇E
        dir = TangentVector(.- grad_vec, info.ζ)
        dir, merge(info, (; dir))
    end
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
    prefix = preconditioned ? "prec_SD" : "SD"
    (;next_dir, name, prefix, preconditioned)
end

@doc raw"""
    OLD: conjugate_gradient(; preconditioned=true, cg_type="Fletcher-Reeves")

(Preconditioned) conjugate gradient algorithm on the MO manifold.
The ``cg_type`` for now is useless but will serve to launch other
types of CG algorithms.
"""
function conjugate_gradient(;preconditioned=true,
                            cg_type=:Fletcher_Reeves,
                            transport_type=:exp)
    function next_dir(info)
        ζ = info.ζ
        ∇E = info.∇E
        ∇E_prev = info.∇E_prev
        dir = info.dir
        # DEBUG : The prec gradient is bad.
        # foo = preconditioned_gradient_AMO(ζ)
        # @show foo[2]            
        current_grad = preconditioned ? preconditioned_gradient_AMO(ζ)[1] : ∇E
        # @show test_tangent(TangentVector(current_grad, ζ))
        # Transport previous dir and gradient on current point ζ
        τ_dir_prev = transport_AMO(dir, dir.base, dir, 1., ζ; type=transport_type, collinear=true)
        # @show test_tangent(τ_dir_prev)
 
        # Assemble CG dir with Fletcher-Reeves or Polack-Ribiere coefficient
        cg_factor = begin
            if cg_type==:Fletcher_Reeves
                τ_grad_prev = transport_AMO(∇E_prev, ∇E_prev.base, dir, 1., ζ; type=transport_type, collinear=false)
                tr(∇E'τ_grad_prev)
            else
                zero(ζ.energy)
            end
        end
        β = (norm(∇E)^2 - cg_factor) / norm(info.∇E_prev)^2
        β = (β > 0) ? β : zero(Float64) # Automatic restart if β_PR < 0

        dir = TangentVector(-current_grad + β*τ_dir_prev, ζ)
        dir, merge(info, (; dir))
    end
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    (;next_dir, name, prefix, preconditioned)
end

cos_angle_vecs(X,Y) = tr(X'Y) / √(tr(X'X)*tr(Y'Y))

function lbfgs(depth=8; B₀=default_LBFGS_init, preconditioned=false)
    function next_dir(info)
        # Extract data
        B = info.B
        x_prev = info.dir.base
        x_new = info.ζ
        ∇E = info.∇E
        ∇E_prev = info.∇E_prev
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
                                                 ).vec, x_new)
        ρ = 1/tr(s'y)
        push!(B, (s,y,ρ))

        # Compute next dir
        dir_vec = -B(∇E; B₀)
        dir = TangentVector(-B(∇E).vec, x_new)
        dir, merge(info, (; dir, B))
    end
    name="LBFGS"
    prefix="LBFGS"
    (; next_dir, depth, name, prefix, preconditioned)
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

function default_LBFGS_init(B::LBFGSInverseHessian, g::TangentVector)
    s, y, ρ = B[B.length]
    γ = tr(s'y)/(norm(y)^2)
    Nb = size(g.vec,1)
    γ*Matrix(I, Nb, Nb)
end
# function Fock_LBFGS_inti(B::LBFGSInverseHessian, g::TangentVector)
#     Fi, Fa = Fock_operators(g.base)
# end

