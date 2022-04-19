function rohf_steepest_descent(ζ::ROHFState;
                               max_iter = 500,
                               max_step = 2*one(Float64), ρ = 0.5,
                               cv_threshold = 1e-5,
                               restart = false,
                               linesearch_type = BackTracking(order=3),
                               )

    (typeof(max_step)≠Float64) && (max_step = Float64(max_step))

    # Transfer in orthonormal AO convention
    orthonormalize_state!(ζ)    
    @info("Conditioning of the overlap: $(cond(ζ.Σ.overlap_matrix))")
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);    

    # Initial data
    E, ∇E = rohf_energy_and_gradient(ζ.Φ, Sm12, ζ)
    E_previous = E; p = .- ∇E
    δE = zero(E); norm_residual = zero(E)
    step = zero(Float64); iter=zero(Int64)
    converged = false

    header = ["Energy","δE", "||Π∇E||", "Step", "Iter"]
    println("-"^65)
    println(@sprintf("%-16s  %-16s  %-16s  %-5s  %-5s", header...))
    println("-"^65)
    flush(stdout)

    while (!converged & (iter < max_iter))
        iter += 1;
        # k -> k+1
        step, E, ζ = rohf_linesearch(ζ, p, Sm12, E = E, ∇E = ∇E,
                           max_step = max_step, linesearch_type = linesearch_type)

        δE = E - E_previous; E_previous = E

        # New gradient and direction
        ∇E = grad_E_MO_metric(ζ.Φ, Sm12, ζ)
        p = .- ∇E

        # Check convergence
        norm_residual = norm(∇E)
        (norm_residual < cv_threshold) && (converged=true)
        
        # Print current info
        infos = [E, δE, norm_residual, step, iter]
        println(@sprintf("%16.12f %16.12f %16.12f  %1.5f %5i", infos...))
        flush(stdout)                 
    end
    (converged)  && println("-"^65*"\n"*"CONVERGED")
    !(converged) && println("----Maximum iteration reached")
    infos = [E, δE, norm_residual, step, iter]
    println(@sprintf("%16.12f %16.12f %16.12f  %1.5f %5i", infos...))
    flush(stdout)

    ζ
end
