function conjugate_gradient_solver(; preconditioned=true, cg_type="Fletcher-Reeves")
    function next_dir(info, Sm12)
        ζ = info.ζ
        current_grad = preconditioned ? preconditioned_gradient(ζ, Sm12) : .- info.∇E

        # When CG is restarted (or for first iter) the chosen dir is simply
        # opposite to the preconditioned gradient
        if info.cg_restart
            dir =  ROHFTangentVector(.- current_grad, info.ζ)
            return dir, merge(info, (;dir=dir, cg_restart=false))
        end
        
        # Transport previous dir on point ζ
        τ_dir_prev = transport_vec_along_himself(info.dir, 1., ζ).vec

        # Assemble new CG dir
        dir = info.dir
        β = norm(info.∇E)^2 / norm(grad_E_MO_metric(dir.foot.Φ, Sm12, dir.foot))^2
        dir = ROHFTangentVector(.- current_grad .+ β .* τ_dir_prev, ζ)
        # @show (norm(dir.vec .- proj_horizontal_tangent_space(ζ.Φ, dir.vec, ζ.M.mo_numbers)))

        # Check restart conditions.
        # The first condition checks that the direction is a descent direction
        # The second one is from experiment and insures seldom forced restarts
        # to avoid convergence plateaux.
        Nb, Nd, Ns = ζ.M.mo_numbers
        No = Nd+Ns
        cos_angle_dir_∇E = cos_angle_vecs(dir.vec, .- info.∇E)
        if ((cos_angle_dir_∇E < 0.1) | (info.n_iter%(No*(Nb-No))==0))
            return dir, merge(info, (;cg_restart=true))
        end
        dir, merge(info, (;dir=dir))        
    end
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    (;next_dir, name, preconditioned)
end

cos_angle_vecs(X,Y) = tr(X'Y) / √(tr(X'X)*tr(Y'Y))
