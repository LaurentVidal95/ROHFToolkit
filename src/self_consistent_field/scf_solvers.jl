# SCF with no diis
function SCF_DIIS(info;
                  diis=DIIS(;m=15),
                  callback=SCF_default_callback(),
                  maxiter=500,
                  )

    # Standard ROHF SCF fix point map
    function fixpoint_map(info; diis)
        n_iter, converged = info.n_iter, info.converged
        converged && return info # end process if converged
        n_iter += 1

        # Compute current densities (or DIIS extrapolation if activated)
        ζ = info.ζ
        Pd, Ps = diis(info)

        # Assemble effective Hamiltonian
        Fd, Fs = Fock_operators(Pd, Ps, ζ)
        H_eff = assemble_H_eff(H_eff_coeffs(info.effective_hamiltonian, ζ.Σ.mol)...,
                               Pd, Ps, Fd, Fs)
        # Compute new DMs with aufbau
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:size(ζ.Φ,1)]
        ζ.Φ = Φ_out
        Pd_out, Ps_out = densities(ζ)

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

    # Initial print
    callback(info)

    # Actual SCF DIIS loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; diis)
        @assert (test_MOs(info.ζ) < 1e-10)
    end
    info
end

# TODO
function oda()
    nothing
end

# TODO
function g_new_diis()
    nothing
end
