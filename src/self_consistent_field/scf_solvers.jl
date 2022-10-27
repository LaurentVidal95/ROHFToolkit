# SCF with no diis
function scf_diis(info;
                  diis=DIIS(;m=15),
                  callback=SCF_default_callback(),
                  maxiter=500,
                  )
    (diis.m==0) && (@warn "Beware: DIIS is inactive")

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
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:size(ζ.Φ,2)]
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

function hybrid_SCF_optimization_args(ζ₀::ROHFState)
    Pd₀, Ps₀ = densities(ζ₀)
    Fd₀, Fs₀ = Fock_operators(Pd₀, Ps₀, ζ₀)

    # Initial point
    ζ_init = ζ₀
    # H_eff = assemble_H_eff(H_eff_coeffs(guess, ζ₀.Σ.mol)...,
    #                        Pd, Ps, Fd, Fd)
    # Φ_init = eigvecs(Symmetric(H_eff))[:,1:size(Φ₀,2)]
    # ζ_init = ROHFState(ζ₀, Φ_init)

    # Function to minimize
    function fg₀(ζ::ROHFState)
        Φd, Φs = split_MOs(ζ)
        f = tr(Fd₀*Φd*Φd') + tr(Fs₀*Φs*Φs')
        g₀_vec = project_tangent(ζ.Σ.mo_numbers, ζ.Φ, hcat(2*Fd₀*Φd, 2*Fs₀*Φs))
        f, ROHFTangentVector(g₀_vec, ζ)
    end
    fg₀, ζ_init
end

function hybrid_scf(info;
                    diis=DIIS(;m=15),
                    callback=SCF_default_callback(),
                    maxiter=500,
                    )

    (diis.m==0) && (@warn "Beware: DIIS is inactive")

    # Standard ROHF SCF fix point map
    function fixpoint_map(info; diis)
        n_iter, converged = info.n_iter, info.converged
        converged && return info # end process if converged
        n_iter += 1

        # Solve subproblem with direct minimization on the ROHF manifold
        fg, ζ_init = hybrid_SCF_optimization_args(info.ζ)
        ζ, _, _, history = optimize(fg, ζ_init,
                                    ConjugateGradient(;verbosity=0, gradtol=1e-6);
                                    optim_kwargs(;preconditioned=false, verbose=false)...)
        
        # Compute new energy and residual
        E, ∇E = energy_and_gradient(ζ)
        E_prev = info.E
        residual = norm(∇E)
        (residual < info.tol) && (converged = true)

        info = merge(info, (; ζ=ζ, n_iter=n_iter, E=E, ∇E=∇E, E_prev=E_prev,
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
