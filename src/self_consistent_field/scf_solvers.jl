#
# Standard SCF with DIIS by default.
#
function scf(info; fixpoint_map, maxiter=500)
    # Define standard scf update function
    function g_std(Pd, Ps, Fd, Fs, ζ::ROHFState, info)
        # Assemble effective Hamiltonian
        H_eff = assemble_H_eff(H_eff_coeffs(info.effective_hamiltonian, ζ.Σ.mol)...,
                               Pd, Ps, Fd, Fs)
        # Compute new DMs with aufbau
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:size(ζ.Φ,2)]
        ζ.Φ = Φ_out
        ζ, densities(ζ)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_std)
        @assert(test_MOs(info.ζ) < 1e-10)
    end
    info
end

"""
Choose next densities in the argmin of the SCF iteration problem.
Use to be called g_new diis
"""
function hybrid_scf(info; fixpoint_map, maxiter=500)
    # Define hybrid scf update function
    function g_hybrid(Pd, Ps, Fd, Fs, ζ::ROHFState, info;
                      guess=info.effective_hamiltonian)
        fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ, guess)

        function hybrid_SCF_precondition(ζ::ROHFState, η)
            prec_grad = ROHFTangentVector(hybrid_SCF_preconditioned_grad(Pd, Ps, ζ), ζ)
            (tr(prec_grad'η) ≤ 0) && (@warn "no preconditioning"; return η)
            prec_grad
        end

        # Solve subproblem with CG
        kwargs = merge(optim_kwargs(;preconditioned=false, verbose=false),
                       (; precondition=hybrid_SCF_precondition))
        
        ζ, _, _, history = optimize(fg, ζ_init,
                           ConjugateGradient(;verbosity=1, gradtol=1e-8);
                           kwargs...)
        ζ, densities(ζ)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_hybrid)
        @assert(test_MOs(info.ζ) < 1e-10)
    end
    info
end
function hybrid_SCF_optimization_args(Pd₀::Matrix{T}, Ps₀::Matrix{T},
                                      Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                      ζ₀::ROHFState{T},
                                      guess::Symbol) where {T<:Real}
    # Initial point
    ζ_init = ζ₀
    if (guess ≠ :none)
        H_eff = assemble_H_eff(H_eff_coeffs(guess, ζ₀.Σ.mol)...,
                               Pd₀, Ps₀, Fd₀, Fs₀)
        Φ_init = eigvecs(Symmetric(H_eff))[:,1:size(ζ₀.Φ,2)]
        ζ_init = ROHFState(ζ₀, Φ_init)
    end

    # Function to minimize
    function fg₀(ζ::ROHFState)
        Φd, Φs = split_MOs(ζ)
        f = tr(Fd₀*Φd*Φd') + tr(Fs₀*Φs*Φs')
        g₀_vec = project_tangent(ζ.Σ.mo_numbers, ζ.Φ, hcat(2*Fd₀*Φd, 2*Fs₀*Φs))
        f, ROHFTangentVector(g₀_vec, ζ)
    end    
    fg₀, ζ_init
end
# Same as classic preconditioner but with Fd = (1/2)*Fd₀ and Fs (1/2)*Fs₀
function hybrid_SCF_preconditioned_grad(Pd₀::Matrix{T}, Ps₀::Matrix{T},
                                        ζ::ROHFState{T}) where {T<:Real}
    Fd₀, Fs₀ = Fock_operators(Pd₀, Ps₀, ζ)
    preconditioned_gradient_MO_metric((1/2) .* Fd₀, (1/2) .* Fs₀, ζ)
end
