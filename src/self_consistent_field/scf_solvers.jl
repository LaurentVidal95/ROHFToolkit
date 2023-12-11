@doc raw"""
     scf(info; fixpoint_map, maxiter=500)

Standard SCF iteration. The SCF depends on the choice of effective ROHF
Hamiltonian (see ``src/self_consistent_field/effective_hamiltonians.jl``).  
"""
function scf(info; fixpoint_map, maxiter=500)
    # Define standard scf update function
    function g_std(Pd, Ps, Fd, Fs, ζ::ROHFState, info)
        # Assemble effective Hamiltonian
        H_eff = assemble_H_eff(H_eff_coeffs(info.effective_hamiltonian, ζ.Σ.mol)...,
                               Pd, Ps, Fd, Fs)
        # Compute new DMs with aufbau
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:size(ζ.Φ,2)]
        ζ_out = ROHFState(Φ_out, ζ.Σ, ζ.energy, ζ.isortho, ζ.guess, ζ.history)
        ζ_out, densities(ζ_out)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_std)
        @assert(test_MOs(info.ζ) < 1e-8)
    end
    info
end

@doc raw"""
    hybrid_scf(info; fixpoint_map, maxiter=500)

Hybrid SCF solver, where the next density is given by:
```math
    g(xₙ) = {\rm argmin}\left\{ {\rm Tr}\left(F_d^{n}P_d+F_s^{n}P_s\right),
            \quad (P_d,P_s)\in\mathcal{M}_{\rm DM}\right\}
```
which is solved by a direct minimization method.
"""
function hybrid_scf(info; fixpoint_map, maxiter=500, inner_loop_verbosity=0)
    # Define hybrid scf update function
    function g_hybrid(Pd, Ps, Fd, Fs, ζ::ROHFState, info;
                      guess=info.effective_hamiltonian)
        fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ, guess)

        function hybrid_SCF_precondition(ζ::ROHFState, η)
            prec_grad = ROHFTangentVector(hybrid_SCF_preconditioned_grad(Pd, Ps, ζ), ζ)
            # @show (norm(η)/norm(prec_grad))
            # @show tr(prec_grad'η)/(norm(η)*norm(prec_grad))
            if (tr(prec_grad'η)/(norm(prec_grad)*norm(η))) ≤ 1e-5
                @warn "No preconditioning"
                return η
            end
            prec_grad
        end

        # Solve subproblem with CG
        kwargs = merge(optim_kwargs(;preconditioned=false, verbose=false),
                       (; precondition=hybrid_SCF_precondition))

        ζ, _, _, history = optimize(fg, ζ_init,
                           GradientDescent(;verbosity=inner_loop_verbosity, gradtol=1e-7);
                           kwargs...)
        # Problem... Ne pas faire un restart avec le ζ juste au dessus.
        if (test_MOs(ζ) ≥ 1e-9)
            @warn "Trying new guess"
            fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ, :Euler)
            ζ, _, _, history = optimize(fg, ζ_init,
                                        GradientDescent(;verbosity=inner_loop_verbosity, gradtol=1e-7);
                                        kwargs...)
        end
        ζ, densities(ζ)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_hybrid)
        @assert(test_MOs(info.ζ) < 1e-8)
    end
    info
end
@doc raw"""
    hybrid_SCF_optimization_args(Pd₀::Matrix{T}, Ps₀::Matrix{T},
                                      Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                      ζ₀::ROHFState{T},
                                      guess::Symbol) where {T<:Real}

Fixes the arguments of the Riemannian optimization problem of the
hybrid SCF in OptimKit convention.
"""
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
