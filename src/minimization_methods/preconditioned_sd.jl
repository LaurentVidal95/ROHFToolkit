using IterativeSolvers

"""
    One iteration of preconditioned steepest descent algorithm
"""
function preconditioned_SD_step(ΦT, Sm12, A, H, N_bds;
                                E_previous = zero(Float64),
                                atom_info = [],
                                linesearch_type = BackTracking(order=3),
                                max_step = one(Float64),
                                numerical_safety = 0.1,
                                )
 
    Nb,Nd,Ns = N_bds
    No = Nd + Ns; Nv = Nb - No
    Nn = Nd*Ns + Nd*Nv + Ns*Nv

    ΦdT, ΦsT = split_MOs(ΦT, N_bds);

    ################################################################################
    ##                         X,Yv,Zv as a vector convention                     ##
    ################################################################################
    FdT, FsT = compute_Fock_operators(ΦT, Sm12, A, H, N_bds)
    
    # L, b = build_prec_grad_system_MOs(ΦdT, ΦsT, FdT, FsT, N_bds) 
    L, b = build_prec_grad_system_MOs(ΦdT, ΦsT, FdT, FsT, N_bds, numerical_safety = numerical_safety)
    
    # Solve the preconditionning linear system
    XYvZv = bicgstabl(L, -b, 3, reltol = 1e-14, abstol = 1e-14)

    ################################################################################
    ##                         Back in (Ψd, Ψs) convention                        ##
    ################################################################################

    X, Yv, Zv = vec_to_mat_MO(XYvZv, N_bds) # Preconditioned gradient
    p_dir = ΦsT*X' + Yv, -ΦdT*X + Zv

    ########### NUMERICAL FIX
    test = norm(p_dir .- proj_horizontal_tangent_space(ΦT, p_dir, N_bds))
    if( test > 1e-8 )
        println("Rasing numerical safety")
        flush(stdout)
        L, b = build_prec_grad_system_MOs(ΦdT, ΦsT, FdT, FsT, N_bds, numerical_safety = 1e-2)
        XYvZv = bicgstabl(L, -b, 3, reltol = 1e-14, abstol = 1e-14)
        X, Yv, Zv = vec_to_mat_MO(XYvZv, N_bds) # Preconditioned gradient                                                                                                                                         
        p_dir = ΦsT*X' + Yv, -ΦdT*X + Zv
        @show(norm(p_dir .- proj_horizontal_tangent_space(ΦT, p_dir, N_bds)))
    end
    
    # Purify descent direction
    p_dir = proj_horizontal_tangent_space(ΦT, p_dir, N_bds)
    ################# NUMERICAL FIX
    
    ∇E_dir = proj_horizontal_tangent_space(ΦT, (4*FdT*ΦdT, 4*FsT*ΦsT), N_bds)
  
    # Test: is p a descent dir ?
    cos_p_∇E = scal_M(∇E_dir,p_dir)/(
        √(scal_M(∇E_dir,∇E_dir)*scal_M(p_dir,p_dir)) )

    # linesearch
    step = zero(Int64); E = zero(Int64); ΦT_next = similar(ΦT);
    
    # Do manual armijo linesearch to avoid bugs when dir is almost not a descent dir.
    if (cos_p_∇E ≥ -0.05)
        println("Manual Armijo LS")
        flush(stdout)
        
        step, E, ΦT_next = manual_line_search(ΦT, p_dir, Sm12, A, H, N_bds,
                                              atom_info=atom_info,E = E_previous,
                                              ∇E = ∇E_dir, max_step = max_step,)
    else
        step, E, ΦT_next = line_search(ΦT, p_dir, Sm12, A, H, N_bds,
                                       atom_info=atom_info, E = E_previous,
                                       ∇E = ∇E_dir, max_step = max_step,
                                       linesearch_type = linesearch_type)
    end
    
    ΦT_next, E, norm(∇E_dir), cos_p_∇E, step
end



"""
    Preocnditionned steepest descent on the ROHF manifold.
    The preconditionner is one approximation of the hessian matrix given in 
    [insert doi article] formula (27).
    
    Parameters are

    <><> General
    - Σ: the chemical system of the computation
    - max_iter: maximum number of preconditioned SD iterations
    - thresh_E & thresh_Π∇E: convergence threshold on the δE and norm of the projected gradient.
    Both are 0 on a minimum.

    <><> Print and save data
    - verbosity ∈ ("high", "medium", "head", "tail")
        -> high prints every informations
        -> head prints every thing except the tail
        -> tail doesn't print the head 
        -> medium only print the iterations without head or tail.
    - save_MOs & MOs_dir: if true, save the MOs at each iterations in the dir MOs_dir/
        Usefull to be able to exit stop a computation without any data loss.
    - Σ_name: name of the system appearing in the MOs files generated by save_MOs

    <><> Starting MOs
    - Φ_init: initial MOs for the computation. 
      If none are given, starting MOs are the one stored in Σ.
    - restart: if true, set the orbitals stored in Σ as the core hamiltonian guess/
""" 
function rohf_preconditioned_SD(Σ::ChemicalSystem;
                                Φ_init = zero(Σ.overlap_matrix),
                                ΦT_min = zero(Σ.overlap_matrix),
                                max_iter = 500,
                                max_step = one(Float64),
                                linesearch_type = BackTracking(order=3),
                                cv_threshold= 1e-5,
                                verbosity = "high", save_MOs = false,
                                MOs_dir = "out",
                                Σ_name = "preconditioned_SD",
                                restart = false,
                                numerical_safety = 0.1)

    (typeof(max_step)≠Float64) && (max_step = Float64(max_step))
    
    (cv_threshold > 1e-4) && @warn("Warning: the convergence threshold is low and"*
                                   " will induce lower precision.")
    tic = now()
    @assert( verbosity ∈ ("high", "head", "tail", "medium", "none") )
    
    # Write head
    if( verbosity ∈ ("high", "head") )
        println("Computation started at $(tic)"*"\n")
        println("Minimization of the energy on the ROHF manifold")
        (verbosity =="high") && (println("Parameters: max_iter = $max_iter, linesearch = $(linesearch_type)"))
        (verbosity =="high") &&
            (println("Stopping criteria: ||Π∇E|| < $(cv_threshold)"*"\n"));
        flush(stdout);
    end

    # Extract system informations
    N_bds, A, S, H, atom_info = read_system(Σ)
    S12 = sqrt(Symmetric(S)); Sm12=inv(S12);

    # Extract initial MOs from Σ and orthogonalise.
    if( iszero(Φ_init) )
        (restart) && (reset_system!(Σ))
        Φ_init = Σ.MOs
    end
    (restart) && (reset_system!(Σ))
    ΦT = ortho_AO(Φ_init, S12)
    
    if save_MOs
        !(isdir("$(MOs_dir)")) && (mkdir("$(MOs_dir)"))
    end
    
    # Initial gradient and direction p
    E = energy_from_MOs_T(ΦT, Sm12, A, H, N_bds, atom_info = atom_info)
    E_previous = E
    δE = zero(E); norm_residual = zero(E)
    step = zero(Float64);  cos_p_∇E = zero(Float64)
    
    iter, computation_time = Σ.iter_and_time
    converged = false
    
    if save_MOs
        !(isdir("$(MOs_dir)")) && (mkdir("$(MOs_dir)"))
    end
    
    # Print initial infos
    if !(verbosity == "none")
        (verbosity ∈ ("high","head")) && (println("Guess energy : $(E_previous) \n"))
        println("-"^85)
        println(@sprintf("%-16s  %-16s  %-16s %-16s %-5s  %-5s","Energy","δE",
                         "||Π∇E||","cos(p,∇E)", "Step", "Iter"))
        
        println("-"^85)
        flush(stdout)
    end
    
    while( !(converged) & (iter < max_iter) )
        iter += 1
        
        ΦT, E, norm_residual, cos_p_∇E, step =  preconditioned_SD_step(ΦT, Sm12, A, H, N_bds,
                                                                       E_previous = E_previous,
                                                                       atom_info = atom_info,
                                                                       max_step = max_step,
                                                                       linesearch_type = linesearch_type,
                                                                       numerical_safety = numerical_safety,)
        δE = E - E_previous
        E_previous = E

        # Test admissibility of MOs
        (test_MOs(ΦT,N_bds) > 1e-8) && (@error("Non admissible densities"))
                      
        # Save_current MOs if needed
        (save_MOs) && (save_MOs_in_file(Sm12*ΦT, "$(MOs_dir)/MOs_$(Σ_name)_$(iter)"; E = E, iter = iter))
       
        if !(verbosity == "none")
            println(@sprintf("%16.12f %16.12f %16.12f %16.12f  %1.5f %5i  | Preconditioned SD",
                             E, δE, norm_residual, cos_p_∇E, step, iter))
            flush(stdout)
        end

        if( norm_residual < cv_threshold )
            converged = true
        end

        # Actualize info in the ChemicalSystem structure
        Φ = Sm12*ΦT # De-orthonomalize the MOs
        Σ.MOs = Φ; Σ.E_rohf = E
        Σ.iter_and_time = (iter, computation_time)
        Σ.cv_history = (converged, norm_residual)
    end

    if( verbosity ∈ ("high","tail") )
        (converged)  && println("-"^85*"\n"*"CONVERGED")
        !(converged) && println("----Maximum iteration reached")
        println(@sprintf("%16.12f %16.12f %16.12f %16.12f  %1.5f %5i",
                         E, δE, norm_residual, cos_p_∇E,  step, iter))
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
    Σ.iter_and_time = (iter, computation_time)
    Σ.cv_history = (converged, norm_residual)

    Σ
end
