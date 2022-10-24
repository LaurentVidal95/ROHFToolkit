# SCF with no diis
function SCF_DIIS(info;
                  diis=DIIS(;m=15),
                  callback=SCF_default_callback(),
                  maxiter=500,
                  )

    # Standard ROHF SCF fix point map
    function fixpoint_map(Φ, info)
        n_iter, converged = info.n_iter, info.converged
        converged && return info # end process if converged
        n_iter += 1

        ζ = info.ζ

        # Assemble effective Hamiltonian
        Nb, Nd, Ns = ζ.M.mo_numbers
        Pd, Ps = densities(Φ, (Nb, Nd, Ns))
        Fd, Fs = compute_Fock_operators(Φ, ζ)

        H_eff = assemble_H_eff(H_eff_coeffs(info.effective_hamiltonian, ζ.Σ.mol)...,
                               Pd, Ps, Fd, Fs)
        # Compute new MOs
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:Nd+Ns]
        ζ.Φ = Φ_out
        
        # Compute new energy and residual
        E_prev = info.E
        E, ∇E = rohf_energy_and_gradient(ζ)
        
        # check for convergence
        residual = norm(∇E)
        (residual < info.tol) && (converged = true)

        info = merge(info, (; ζ=ζ, n_iter=n_iter, E=E, ∇E=∇E, E_prev=E_prev,
                            residual=residual, converged=converged))
        callback(info)

        Φ_out, info
    end

    # Initial print
    callback(info)

    # Actual SCF DIIS loop
    Φ, ∇E = info.ζ.Φ, info.∇E
    while ( (!info.converged) && (info.n_iter < maxiter) )
        Φ_diis = diis(Φ, ∇E; info)
        Φ, info = fixpoint_map(Φ_diis, info)
        ∇E = info.∇E
        @assert (test_MOs(Φ, info.ζ.M.mo_numbers) < 1e-10)
    end

    Φ, info
end

# TODO
function oda()
    nothing
end

# TODO
function g_new_diis()
    nothing
end
