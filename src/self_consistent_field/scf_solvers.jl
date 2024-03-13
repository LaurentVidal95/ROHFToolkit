@doc raw"""
     scf(info; fixpoint_map, maxiter=500)

Standard SCF iteration. The SCF depends on the choice of effective ROHF
Hamiltonian (see ``src/self_consistent_field/effective_hamiltonians.jl``).  
"""
function scf(info; fixpoint_map, maxiter=500)
    # Define standard scf update function
    function g_std(Pd, Ps, Fd, Fs, ζ::State, info)
        # Assemble effective Hamiltonian
        H_eff = assemble_H_eff(H_eff_coeffs(info.effective_hamiltonian, ζ.Σ.mol)...,
                               Pd, Ps, Fd, Fs)
        # Compute new DMs with aufbau
        Φ_out = eigvecs(Symmetric(H_eff))[:,1:size(ζ.Φ,2)]
        ζ_out = State(Φ_out, ζ.Σ, ζ.energy, ζ.isortho, ζ.guess, ζ.virtuals, ζ.history)
        ζ_out, densities(ζ_out)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_std)
        @assert is_point(info.ζ)
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
    function g_hybrid(Pd, Ps, Fd, Fs, ζ::State, info;
                      guess=info.effective_hamiltonian)
        fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ, guess)

        function hybrid_SCF_precondition(ζ::State, η)
            (norm(η) > 10^(-1/2)) && (return η)
            return hybrid_SCF_preconditioned_grad(Pd, Ps, ζ)
        end

        # Solve subproblem with CG
        kwargs = merge(optimkit_kwargs(;preconditioned=false, verbose=false),
                       (; precondition=hybrid_SCF_precondition))

        ζ, _, _, history = optimize(fg, ζ_init,
                                    GradientDescent(;verbosity=inner_loop_verbosity, gradtol=1e-5,
                                                    maxiter=15);
                           kwargs...)
        # Problem... Ne pas faire un restart avec le ζ juste au dessus.
        if !is_point(ζ; tol= 1e-9)
            @warn "Trying new guess"
            fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ, :Euler)
            ζ, _, _, history = optimize(fg, ζ_init,
                                        GradientDescent(;verbosity=inner_loop_verbosity, gradtol=1e-5,
                                                        maxiter=15); kwargs...)
        end
        ζ, densities(ζ)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_hybrid)
        @assert is_point(info.ζ)
    end
    info
end
@doc raw"""
    hybrid_SCF_optimization_args(Pd₀::Matrix{T}, Ps₀::Matrix{T},
                                      Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                      ζ₀::State{T},
                                      guess::Symbol) where {T<:Real}

Fixes the arguments of the Riemannian optimization problem of the
hybrid SCF in OptimKit convention.
"""
function hybrid_SCF_optimization_args(Pi₀::Matrix{T}, Pa₀::Matrix{T},
                                      Fi₀::Matrix{T}, Fa₀::Matrix{T},
                                      ζ₀::State{T},
                                      guess::Symbol) where {T<:Real}
    # Initial point
    ζ_init = ζ₀
    if (guess ≠ :none)
        H_eff = assemble_H_eff(H_eff_coeffs(guess, ζ₀.Σ.mol)...,
                               Pi₀, Pa₀, Fi₀, Fa₀)
        Φ_init = eigvecs(Symmetric(H_eff))[:,1:size(ζ₀.Φ,2)]
        ζ_init = State(ζ₀, Φ_init)
    end

    # Function to minimize
    function fg₀(ζ::State)
        Φi, Φa, Φe = split_MOs(ζ.Φ, ζ.Σ.mo_numbers; virtuals=true)
        
        f = tr(Fi₀*Φi*Φi') + tr(Fa₀*Φa*Φa')

        Gx = -(1/2)*Φi'*(Fi₀-Fa₀)*Φa
        Gy = -Φi'Fi₀*Φe
        Gz = -Φa'Fa₀*Φe
        # Assemble into kappa matrix
        Nb, Ni, Na = ζ.Σ.mo_numbers
        Ne = Nb-(Ni+Na)
        Gκ = [zeros(Ni,Ni) Gx Gy; -Gx' zeros(Na,Na) Gz; -Gy' -Gz' zeros(Ne,Ne)]
        
        f, TangentVector(Gκ, ζ)
    end    
    fg₀, ζ_init
end
# Same as classic preconditioner but with Fd = (1/2)*Fd₀ and Fs (1/2)*Fs₀
function hybrid_SCF_preconditioned_grad(Pd::Matrix{T}, Ps::Matrix{T},
                                        ζ::State{T}) where {T<:Real}
    Fi, Fa = Fock_operators(Pd, Ps, ζ)
    Fi = (1/2) .* Fi
    Fa = (1/2) .* Fa
    # Construct gradient for hybrid scf problem
    Φi, Φa, Φe = split_MOs(ζ.Φ, ζ.Σ.mo_numbers; virtuals=true)
    Gx = -Φi'*(Fi-Fa)*Φa
    Gy = -2*Φi'Fi*Φe
    Gz = -2*Φa'Fa*Φe
    # Assemble into kappa matrix
    Nb, Ni, Na = ζ.Σ.mo_numbers
    Ne = Nb-(Ni+Na)
    Gκ = [zeros(Ni,Ni) Gx Gy; -Gx' zeros(Na,Na) Gz; -Gy' -Gz' zeros(Ne,Ne)]
    AMO_preconditioner(Fi, Fa, TangentVector(Gκ, ζ))
end
