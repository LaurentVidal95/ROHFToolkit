function scf_method(ζ::ROHFState;
                    solver = scf,
                    diis = DIIS(;m=15),
                    effective_hamiltonian=:Guest_Saunders,
                    tol=1e-5,
                    callback=SCF_default_callback(),
                    kwargs...)
    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)

    (diis.m==0) && (@warn "Beware: DIIS is inactive")

    # Populate info with initial data
    n_iter       = zero(Int64)
    DMs          = densities(ζ)
    E, ∇E        = energy_and_gradient_DM_metric(DMs..., ζ)
    E_prev       = NaN
    residual     = norm(∇E) # √(tr(∇Ed'∇Ed) + tr(∇Es'∇Es))
    converged    = (residual < tol)

    info = (; n_iter, ζ, DMs, E, E_prev, ∇E, effective_hamiltonian, converged, tol,
            solver)

    # Fixpoint map common to all SCF routines. g is such that:
    # Pdₙ₊₁, Psₙ₊₁ = g(Pdₙ, Psₙ)
    function fixpoint_map(info; g_update)
        n_iter, converged = info.n_iter, info.converged
        converged && return info # end process if converged
        n_iter += 1

        # Compute current densities (or DIIS extrapolation if activated)
        ζ = info.ζ
        Pd, Ps = diis(info)
        Fd, Fs = Fock_operators(Pd, Ps, ζ)

        # n -> n+1 densities and state
        ζ, Pd_out, Ps_out = g_update(Pd, Ps, Fd, Fs, ζ, info)

        # Compute new energy and residual
        E_prev = info.E
        E, ∇E = energy_and_gradient_DM_metric(Pd_out, Ps_out, ζ)
        
        # check for convergence
        residual = norm(∇E)
        (residual < info.tol) && (converged = true)

        info = merge(info, (; ζ=ζ, n_iter=n_iter, DMs = (Pd_out, Ps_out), E=E, ∇E=∇E, E_prev=E_prev,
                            residual=residual, converged=converged))
        callback(info)
        info
    end
    
    callback(info) # Initial print
    # SCF-type loop. See "scf_solvers.jl" for all implemented solvers.
    info = solver(info; fixpoint_map, kwargs...)

    # Return non-orthonormal state
    ζ = info.ζ
    deorthonormalize_state!(ζ)

    # To be removed. Have to do so that the solver doesn't change ζ in place.
    reset_state!(ζ, guess=ζ.guess)
    info = merge(info, (; ζ=ζ))

    clean(info)
end
