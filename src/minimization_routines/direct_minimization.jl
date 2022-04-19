function minimize_rohf_energy(ζ::ROHFState;
                              solver = "steepest descent",
                              max_iter = 500,
                              max_step = 2*one(Float64), ρ = 0.5,
                              cv_threshold = 1e-5,
                              linesearch_type = BackTracking(order=3),
                              prompt=default_prompt(),
                              )

    (typeof(max_step)≠Float64) && (max_step = Float64(max_step))

    # Transfer in orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);
    orthonormalize_state!(ζ, S12=S12)

    # Populate info with initial data
    n_iter = zero(Int64)
    E, ∇E = rohf_energy_and_gradient(ζ.Φ, Sm12, ζ)
    E_prev = NaN
    dir = .- ∇E
    residual = nothing
    step   = zero(Float64)
    converged = false
    info = (; n_iter, ζ, E, E_prev, ∇E, dir, residual,
            solver, step, converged, cv_threshold)

    # Prompt initial data
    prompt(info)
    
    while (!(info.converged) && (n_iter < max_iter))
        n_iter += 1;
        # k -> k+1
        step, E, ζ = rohf_manifold_linesearch(ζ, dir, Sm12, E = E, ∇E = ∇E,
                          max_step = max_step, linesearch_type = linesearch_type)

        # Actualize info with current iter data
        E_prev = info.E
        ∇E = grad_E_MO_metric(ζ.Φ, Sm12, ζ)
        residual = ∇E
        # Remplacer dir par autre chose pour changer d'algo
        dir = .- ∇E
        (norm(residual)<cv_threshold) && (converged=true)
        info = merge(info, (;ζ=ζ, E=E, E_prev=E_prev, ∇E=∇E, dir=dir,
                            residual=residual, converged=converged,
                            n_iter=n_iter, step=step))
        prompt(info)
    end
    deorthonormalize_state!(ζ, Sm12=Sm12)
    info = merge(info, (;ζ=ζ))
    (info.converged)  && println("CONVERGED")
    !(info.converged) && println("----Maximum iteration reached")
    info
end
