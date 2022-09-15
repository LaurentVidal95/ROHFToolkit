function scf_rohf(ζ::ROHFState;
                  max_iter=500,
                  effective_hamiltonian=:Roothan,
                  solver = scf_nlsolve_solver(),
                  tol=1e-5,
                  prompt=default_scf_prompt(),
                  savefile="")

    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);
    orthonormalize_state!(ζ; S12)

    # Populate info with initial data
    n_iter       = zero(Int64)
    E            = rohf_energy!(ζ, Sm12)
    E_prev       = NaN
    Φ            = ζ.Φ
    converged    = false
    residual = Inf

    info = (; n_iter, ζ, E, residual, effective_hamiltonian, converged, tol)

    # Display header and initial data
    prompt.prompt(info)

    function fixpoint_map(Φ_in)
        converged && return Φ_in # end process if converged
        n_iter += 1

        # Assemble effective Hamiltonian
        Nb, Nd, Ns = ζ.M.mo_numbers
        Pd, Ps = densities(Φ_in, (Nb, Nd, Ns))
        Fd, Fs = compute_Fock_operators(Φ_in, Sm12, ζ)

        H_eff = assemble_H_eff(H_eff_coeffs(effective_hamiltonian, ζ.Σ.mol)..., Pd, Ps, Fd, Fs)
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:Nd+Ns]
        
        # Check Aufbau
        # (λ[Nd] ≥ λ[Nd+1]) && @warn("Warning: no aufbau between ds")
        # (λ[Ns] ≥ λ[Ns+1]) && @warn("Warning: no aufbau between sv")

        # Actualize data and check for convergence
        ζ.Φ = Φ_out[:,1:Nd+Ns]
        residual = norm(Φ_out - Φ_in)
        E = rohf_energy!(ζ, Sm12)
        (residual < info.tol) && (converged = true)
        E_prev = info.E

        info = merge(info, (; n_iter=n_iter, ζ=ζ, E=E, E_prev=E_prev,
                            residual=residual, converged=converged))
        prompt.prompt(info)

        Φ_out
    end

    # SCF loop through nlsolve
    Φ = solver(fixpoint_map, Φ, max_iter; tol=eps(eltype(Φ)))
    residual = norm(ζ.Φ - Φ)
    ζ.Φ = Φ
    deorthonormalize_state!(ζ; Sm12)
    E = rohf_energy!(ζ)

    info = merge(info, (; ζ=ζ, E=E, residual=residual))
    prompt.prompt(info)

    (info.converged) ? println("CONVERGED") : println("----Maximum interation reached")

    prompt.clean(info)
end
