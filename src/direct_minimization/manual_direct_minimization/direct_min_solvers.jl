@doc raw"""
    OLD: steepest_descent(; preconditioned=true)

(Preconditioned) Steepest descent algorithm on the MO manifold.
"""
function steepest_descent(; preconditioned=true)
    function next_dir(info)
        grad = preconditioned ? - preconditioned_gradient(info.ζ) : - info.∇E
        dir = TangentVector(grad, info.ζ)
        dir, merge(info,(;dir=dir))
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

function conjugate_gradient(; preconditioned=true, cg_type="Fletcher-Reeves")
    function next_dir(info)
        ζ = info.ζ
        ∇E = info.∇E
        current_grad = preconditioned ? preconditioned_gradient(ζ) : ∇E

        # Transport previous dir and gradient on current point ζ
        τ_dir_prev = transport_vec_along_himself(info.dir, 1., ζ)
        τ_grad_prev = project_tangent(ζ.M, ζ.Φ, info.∇E_prev.vec)
 
        # Assemble CG dir with Polack-Ribière coefficient
        β_PR = (norm(∇E)^2 .- tr(∇E'τ_grad_prev)) / norm(info.∇E_prev)^2
        β = (β_PR > 0) ? β_PR : zero(Float64) # Automatic restart if β_PR < 0

        dir = TangentVector(-current_grad + β * τ_dir_prev, ζ)
        dir, merge(info, (;dir=dir))
    end
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    (;next_dir, name, prefix, preconditioned)
end

cos_angle_vecs(X,Y) = tr(X'Y) / √(tr(X'X)*tr(Y'Y))
