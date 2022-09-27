function steepest_descent(; preconditioned=true)
    function next_dir(info, Sm12)
        grad = preconditioned ? - preconditioned_gradient(info.ζ, Sm12) : - info.∇E
        dir = ROHFTangentVector(grad, info.ζ)
        dir, merge(info,(;dir=dir))
    end
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
    prefix = preconditioned ? "prec_SD" : "SD"
    (;next_dir, name, prefix, preconditioned)
end

function conjugate_gradient(; preconditioned=true, cg_type="Fletcher-Reeves")
    function next_dir(info, Sm12)
        ζ = info.ζ
        ∇E = info.∇E
        current_grad = preconditioned ? preconditioned_gradient(ζ, Sm12) : ∇E

        # Transport previous dir and gradient on current point ζ
        τ_dir_prev = transport_vec_along_himself(info.dir, 1., ζ)
        τ_grad_prev = project_tangent(ζ.M, ζ.Φ, info.∇E_prev.vec)
 
        # Assemble CG dir with Polack-Ribière coefficient
        β_PR = (norm(∇E)^2 .- tr(∇E'τ_grad_prev)) / norm(info.∇E_prev)^2
        β = (β_PR > 0) ? β_PR : zero(Float64) # Automatic restart if β_PR < 0

        dir = ROHFTangentVector(-current_grad + β * τ_dir_prev, ζ)
        dir, merge(info, (;dir=dir))
    end
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    (;next_dir, name, prefix, preconditioned)
end

cos_angle_vecs(X,Y) = tr(X'Y) / √(tr(X'X)*tr(Y'Y))

# function BFGS(; preconditioned=true)
#     function  next_dir(info, Sm12)
#         ζ = info.ζ
#         ∇E = preconditioned ? preconditioned_gradient(ζ, Sm12) : ∇E
#         ∇E_prev = info.∇E_prev
#         dir = info.dir
#         # Previous approximation of Hessian
#         B = info.B
        
#         # Transport previous dir and previous grad
#         s = transport_vec_along_himself(info.dir, 1., ζ)
#         y = ∇E - transport_vec_along_himself(∇E_prev, 1., ζ)

#     end
# end
