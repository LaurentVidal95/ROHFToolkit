using DelimitedFiles

"""
ADD DOC: General direct minimization procedure.
In two words:
1) Choose dir according to the method provided in the solver arg.
2) Linesearch along dir
3) Check convergence
4) Change Delimited files by JSON3, to save more data.
"""
function direct_minimization(ζ::ROHFState;
                             max_iter = 500,
                             max_step = 2*one(Float64),
                             solver = steepest_descent(), # preconditioned
                             tol = 1e-5,
                             linesearch_type = BackTracking(order=3),
                             prompt=default_prompt(),
                             savefile="")
    # Linesearch.jl only handles Float64 step sisze
    (typeof(max_step)≠Float64) && (max_step=Float64(max_step))

    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    S12 = sqrt(Symmetric(ζ.Σ.overlap_matrix)); Sm12=inv(S12);
    orthonormalize_state!(ζ; S12)

    # Populate info with initial data
    n_iter       = zero(Int64)
    E, ∇E        = rohf_energy_and_gradient(ζ.Φ, Sm12, ζ)
    ∇E_prev_norm = norm(∇E)
    E_prev       = NaN
    dir_vec      = solver.preconditioned ? .- preconditioned_gradient(ζ, Sm12) : .- ∇E
    dir          = ROHFTangentVector(dir_vec, ζ)
    step         = zero(Float64)
    converged    = false
    residual = norm(∇E)

    info = (; n_iter, ζ, E, E_prev, ∇E, ∇E_prev_norm, dir, solver, step,
            converged, tol, residual)

    # Display header and initial data
    prompt.prompt(info)

    while (!(info.converged) && (n_iter < max_iter))
        n_iter += 1

        # find next point ζ on ROHF manifold
        step, E, ζ = rohf_manifold_linesearch(ζ, dir.vec, Sm12; E, ∇E, max_step,
                                              linesearch_type)

        # Update "info" with the new ROHF point and related quantities
        ∇E = grad_E_MO_metric(ζ.Φ, Sm12, ζ)
        E_prev = info.E
        residual = norm(info.∇E)
        (residual<tol) && (converged=true)

        info = merge(info, (; ζ=ζ, E=E, E_prev=E_prev, ∇E = ∇E, residual=residual,
                            n_iter=n_iter, step=step, converged=converged))
        prompt.prompt(info)

        # Choose next dir according to the solver and update info with new dir
        dir, info = solver.next_dir(info, Sm12)

        # Save MOs in file if needed
        !isempty(savefile) && (writedlm("rohf_MOs_$(solver.prefix)_current_iter.dat", Sm12*ζ.Φ))
    end
    # Go back to non-orthonormal AO convention
    deorthonormalize_state!(ζ; Sm12)
    info = merge(info, (;ζ=ζ))

    (info.converged) ? println("CONVERGED") : println("----Maximum iteration reached")

    prompt.clean(info)
end
