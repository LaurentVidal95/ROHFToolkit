function self_consistent_field(ζ::ROHFState;
                               max_iter=500,
                               effective_hamiltonian=:Roothan,
                               solver = scf_diis(),
                               tol=1e-5,
                               prompt=default_prompt(),
                               savefile="")
    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);
    orthonormalize_state!(ζ; S12)

    # Populate info with initial data
    n_iter       = zero(Int64)
    E, ∇E        = rohf_energy_and_gradient(ζ.Φ, Sm12, ζ)
    E_prev       = NaN
    residual     = norm(∇E)
    converged    = (residual < tol)
    
    info = (; n_iter, ζ, E, E_prev, ∇E, effective_hamiltonian, converged, tol, solver)

    # Display header and initial data
    prompt.prompt(info)

    function fixpoint_map(Φ)
        converged && return Φ # end process if converged
        n_iter += 1
        
        # Assemble effective Hamiltonian
        Nb, Nd, Ns = ζ.M.mo_numbers
        Pd, Ps = densities(Φ, (Nb, Nd, Ns))
        Fd, Fs = compute_Fock_operators(Φ, Sm12, ζ)

        H_eff = assemble_H_eff(H_eff_coeffs(effective_hamiltonian, ζ.Σ.mol)..., Pd, Ps, Fd, Fs)
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:Nd+Ns]
        ζ.Φ = Φ_out

        # Actualize data
        E_prev = info.E
        E = rohf_energy!(ζ, Sm12)
        Φd, Φs = split_MOs(ζ)
        ∇E = project_tangent(ζ.M, Φ_out, hcat(4Fd*Φd, 4Fs*Φs))

        # check for convergence
        residual = norm(∇E)
        (residual < info.tol) && (converged = true)

        info = merge(info, (; n_iter=n_iter, ζ=ζ, E=E, E_prev=E_prev,
                            residual=residual, converged=converged))
        prompt.prompt(info)

        Φ_out
    end

    # SCF loop through nlsolve
    Φ_out = solver.solve(fixpoint_map, ζ.Φ, max_iter; tol=eps(eltype(ζ.Φ)))[1]
    deorthonormalize_state!(ζ; Sm12)
    (info.converged) ? println("CONVERGED") : println("----Maximum interation reached")

    prompt.clean(info)
end
