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
    function g_hybrid(Pd, Ps, Fd, Fs, ζ::ROHFState, info)
        fg, ζ_init = hybrid_SCF_optimization_args(Pd, Ps, Fd, Fs, ζ)

        function hybrid_SCF_precondition(ζ::ROHFState, η)
            prec_grad = ROHFTangentVector(hybrid_SCF_preconditioned_grad(Pd, Ps, ζ), ζ)
            (tr(prec_grad'η) ≤ 0) && (@warn "no preconditioning"; return η)
            prec_grad
        end
        # Solve subproblem with CG
        kwargs = merge(optim_kwargs(;preconditioned=false, verbose=false),
                       (; precondition=hybrid_SCF_precondition))
        ζ, _, _, history = optimize(fg, ζ_init,
                           ConjugateGradient(;verbosity=1, gradtol=1e-6);
                           kwargs...)
        ζ, densities(ζ)...
    end
    
    # Actual loop
    while ( (!info.converged) && (info.n_iter < maxiter) )
        info = fixpoint_map(info; g_update=g_hybrid)
        @assert (test_MOs(info.ζ) < 1e-10)
    end
    info
end
function hybrid_SCF_optimization_args(Pd₀::Matrix{T}, Ps₀::Matrix{T},
                                      Fd₀::Matrix{T}, Fs₀::Matrix{T},
                                      ζ₀::ROHFState{T}) where {T<:Real}
    # Initial point
    ζ_init = ζ₀
    H_eff = assemble_H_eff(H_eff_coeffs(:Guest_Saunders, ζ₀.Σ.mol)...,
                           Pd₀, Ps₀, Fd₀, Fs₀)
    Φ_init = eigvecs(Symmetric(H_eff))[:,1:size(ζ₀.Φ,2)]
    ζ_init = ROHFState(ζ₀, Φ_init)

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

#
# Relaxed constrained SCF
#
function oad()

end

"""
Find the t_min ∈ [0,1] that minimizes the polynom c1*x^2 + c2*x + c3 
"""
function convex_combination_coeff(c1::T,c2::T,c3::T) where {T<:Real}
    t_min = zero(Int64)
    t_extremum = -c2/(2*c1) # t that gives the global min or max of the polynom
    if c1 > 0
        (t_extremum ≤ 0) && (t_min = zero(T))
        (t_extremum ≥ 1) && (t_min = one(T))
        (0 < t_extremum < 1) && (t_min = t_extremum)
    elseif c1 == 0
        (c2 > 0) && (t_min = zero(T))
        (c2 < 0) && (t_min = one(T))
    else #c1 < 0
        (t_extremum < 0.5) && (t_min = one(T))
        (t_extremum > 0.5) && (t_min = zero(T))
        (t_extremum == 0.5) && (t_min = t_extremum)
    end
    t_min    
end
"""
Build c₁, c₂, c₃ such that
  E((1-t)(Pdₙᶜ,Psₙᶜ) + t(Pdₙ₊₁,Psₙ₊₁)) = c₁*t² + c₂*t + c₃.
We have:
  c₁ = (Pdₙ₊₁-Pdₙᶜ, Psₙ₊₁-Psₙᶜ) ⋅ (Fdₙ₊₁-Fdₙᶜ, Fsₙ₊₁-Fsₙᶜ)
  c₂ = 2(Pdₙ₊₁-Pdₙᶜ, Psₙ₊₁-Psₙᶜ) ⋅ (Fd, Fs)
  c₃ = E(Pdₙᶜ, Psₙᶜ)

Also returns ∇E(Pdₙ₊₁,Psₙ₊₁) to avoid unnecessary computations
"""

function relaxed_constrained_polynom(Pdₙᶜ::Matrix{T}, Psₙᶜ::Matrix{T}, Fdₙᶜ::Matrix{T},
                                     Fsₙᶜ::Matrix{T}, Pdₙ₊₁::Matrix{T}, Psₙ₊₁::Matrix{T},
                                     ζ::ROHFState) where {T<:Real}
    Fdₙ₊₁, Fdₙ₊₁ = Fock_operators(Pdₙ₊₁, Psₙ₊₁, ζ)
    Md =  Pdₙ₊₁ - Pdₙᶜ; Ms = Psₙ₊₁ - Psₙᶜ
    c₁ = tr(Md'(Fdₙ₊₁-Fdₙᶜ) + Ms'(Fsₙ₊₁-Fsₙᶜ))
    c₂ = 2*tr(Md'Fdₙᶜ + Ms'Fsₙᶜ)
    c₃ = energy(Pdₙᶜ, Psₙᶜ, ζ)
    c₁, c₂, c₃ # return Fdₙ₊₁, Fsₙ₊₁ ?
end
