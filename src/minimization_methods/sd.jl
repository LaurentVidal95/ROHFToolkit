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
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);    

    # Initial data
    E, ∇E = rohf_energy_and_gradient(ζ.Φ, Sm12, ζ)
    E_previous = E; p = .- ∇E
    δE = zero(E); norm_residual = zero(E)
    step = zero(Float64)

    header = ["Energy","δE", "||Π∇E||", "Step", "Iter"]
    println("-"^65)
    println(@sprintf("%-16s  %-16s  %-16s  %-5s  %-5s", hearder...))
    println("-"^65)
    flush(stdout)

    ΦT = ROHFTangentVector(ζ)
    
    while (!converged & (iter < max_iter))
        iter += 1;
        # k -> k+1
        step, E, ΦT = line_search(ΦT, p, Sm12, A, H, N_bds,
                                 atom_info = atom_info,
                                 E = E, ∇E = ∇E,
                                 max_step = max_step, ρ = ρ,
                                 linesearch_type = linesearch_type)
                                         

        @assert( test_MOs(ΦT, N_bds) < 1e-10 )

        δE = E - E_previous
        E_previous = E

        # New gradient and direction
        ∇E = grad_E_MO_metric(ΦT, Sm12, A, H, N_bds)
        p = .- ∇E
        norm_residual = norm(∇E)
        
        if( norm_residual < cv_threshold )
            converged = true
        end
        
        # Save current MOs if needed
        (save_MOs) && (save_MOs_in_file(Sm12*ΦT, "$(MOs_dir)/MOs_$(Σ_name)_$(iter)";
                                        E = E, iter = iter))

        # Print infos at k+1
        if !(verbosity == "none")
            println(@sprintf("%16.12f %16.12f %16.12f  %1.5f %5i  | SD",
                             E, δE, norm_residual, step, iter))
            flush(stdout)
        end

        # Actualize system's infos
        Φ = Sm12*ΦT # De-orthonormalize MOs
        Σ.MOs = Φ; Σ.E_rohf = E
        Σ.iter_and_time = (iter,computation_time)
        Σ.cv_history = (converged, norm_residual)
    end

    if( verbosity ∈ ("high","tail") )
        (converged)  && println("-"^65*"\n"*"CONVERGED")
        !(converged) && println("----Maximum iteration reached")
        println(@sprintf("%16.12f %16.12f %16.12f  %1.5f %5i",E, δE, norm_residual, step, iter))
        flush(stdout)
    end
    
    toc = now()
    computation_time += toc - tic

    if( verbosity ∈ ("high","tail") )
        println("\n"^2*"Process ended at $(toc)")
        println("Computation time: $(computation_time)"*"\n")
        flush(stdout)
    end

    Φ = Sm12*ΦT # De-orthonomalize the result
    
    # Store final data in the chemical system structure
    Σ.MOs = Φ; Σ.E_rohf = E
    Σ.iter_and_time = (iter,computation_time)
    Σ.cv_history = (converged, norm_residual)

    Σ
end
