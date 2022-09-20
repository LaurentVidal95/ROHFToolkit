function steepest_descent(; preconditioned=true)
    function next_dir(info, Sm12)
        grad = preconditioned ? .- preconditioned_gradient(info.ζ, Sm12) : .- info.∇E
        dir = ROHFTangentVector(grad, info.ζ)
        dir, merge(info,(;dir=dir))
    end
    name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
    prefix = preconditioned ? "prec_SD" : "SD"
    (;next_dir, name, prefix, preconditioned)
end

# TODO: check second restart condition before to avoid computation of CG dir
# TODO: Replace Fletcher-Reeves coefficient with Polak-Ribière with automatic
# restart, ie β = max(0, β_PR).
function conjugate_gradient(; preconditioned=true, cg_type="Fletcher-Reeves")
    function next_dir(info, Sm12)
        ζ = info.ζ
        ∇E = info.∇E
        current_grad = preconditioned ? preconditioned_gradient(ζ, Sm12) : ∇E
        
        # Transport previous dir on current point ζ
        # τ_dir_prev = transport_vec_along_himself(info.dir, 1., ζ).vec # use vector transport
        τ_dir_prev = project_tangent(ζ.M, ζ.Φ, info.dir.vec) # use projection

        # Assemble CG dir with Fletcher-Reeves coefficient
        β_FR = norm(∇E)^2 / info.∇E_prev_norm^2
        dir = ROHFTangentVector(.- current_grad .+ β_FR .* τ_dir_prev, ζ)

        # Check restart conditions.
        # - The first condition checks that the direction is a descent direction
        # - The second one is from experiment and ensures seldom forced restarts
        #   to avoid convergence plateaux.
        Nb, Nd, Ns = ζ.M.mo_numbers; No = Nd+Ns
        cos_angle_dir_∇E = cos_angle_vecs(dir.vec, .- ∇E)

        if ((cos_angle_dir_∇E < 1e-3) | (info.n_iter%(No*(Nb-No))==0))
            @show (cos_angle_dir_∇E) # DEBUG
            println("RESTART")
            dir = ROHFTangentVector(.- current_grad, ζ)
            return dir, merge(info, (;dir=dir, ∇E_prev_norm=norm(∇E)))
        end

        dir, merge(info, (;dir=dir, ∇E_prev_norm=norm(∇E)))
    end
    name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
    prefix = preconditioned ? "prec_CG" : "CG"
    (;next_dir, name, prefix, preconditioned)
end

cos_angle_vecs(X,Y) = tr(X'Y) / √(tr(X'X)*tr(Y'Y))
