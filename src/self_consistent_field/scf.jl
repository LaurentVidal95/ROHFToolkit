function self_consistent_field(ζ::ROHFState;
                               solver = SCF_DIIS,
                               effective_hamiltonian=:Guest_Saunders,
                               tol=1e-5,
                               solver_kwargs...)                            
    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)

    # Populate info with initial data
    n_iter       = zero(Int64)
    DMs          = densities(ζ)
    E, ∇E        = energy_and_gradient_DM_metric(DMs..., ζ)
    E_prev       = NaN
    residual     = norm(∇E) # √(tr(∇Ed'∇Ed) + tr(∇Es'∇Es))
    converged    = (residual < tol)

    info = (; n_iter, ζ, DMs, E, E_prev, ∇E, effective_hamiltonian, converged, tol,
            solver)

    # SCF-type loop. See "scf_solvers.jl" for all implemented solvers.
    info = solver(info; solver_kwargs...)

    # Return non-orthonormal state
    ζ = info.ζ
    deorthonormalize_state!(ζ)
    # To be removed. Have to do so that the solver doesn't change ζ in place.
    reset_state!(ζ, guess=ζ.guess)
    info = merge(info, (; ζ=ζ))
    
    (info.converged) ? println("CONVERGED") : println("----Maximum interation reached")
    clean(info)
end
