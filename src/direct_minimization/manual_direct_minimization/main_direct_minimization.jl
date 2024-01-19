@doc raw"""
    OLD: direct_minimization(ζ::State;  maxiter = 500, maxstep = 2*one(Float64),
                           solver = conjugate_gradient(), # preconditioned
                           tol = 1e-5, linesearch_type = HagerZhang(),
                           prompt=default_prompt())

General direct minimization procedure, which decomposes as such:
    1) Choose a direction according to the method provided in the solver arg.
    2) Linesearch along direction
    3) Check convergence
The arguments are 
    - ζ: initial point of the optimization on the MO manifold
    - maxiter: maximum number of iterations
    - maxstep: maximum step size during linesearch
    - solver: optimization algorithm. For now only (preconditioned) steepest descent
    and conjugate_gradient.
    - tol: the convergence is asserted when the gradient norm is bellow tol.
    - linesearch_type: linesearch algorithm used at each iteration.
    - prompt: modify prompt if needed. Default should be fine.
"""
function direct_minimization(ζ::State;
                             maxiter = 500,
                             maxstep = 2*one(Float64),
                             solver = conjugate_gradient(), # preconditioned by default
                             tol = 1e-5,
                             linesearch_type = HagerZhang(),
                             prompt=default_prompt())
    # Linesearch.jl only handles Float64 step sisze
    (typeof(maxstep)≠Float64) && (maxstep=Float64(maxstep))

    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)

    # Populate info with initial data
    n_iter          = zero(Int64)
    E, ∇E           = rohf_energy_and_gradient(ζ.Φ, ζ)
    E_prev, ∇E_prev = E, ∇E
    dir_vec         = solver.preconditioned ? .- preconditioned_gradient(ζ) : - ∇E
    dir             = TangentVector(dir_vec, ζ)
    step            = zero(Float64)
    converged       = false
    residual        = norm(∇E)

    info = (; n_iter, ζ, E, E_prev, ∇E, ∇E_prev, dir, solver, step,
            converged, tol, residual)

    # Display header and initial data
    prompt.prompt(info)

    while (!(info.converged) && (n_iter < maxiter))
        n_iter += 1

        # find next point ζ on ROHF manifold
        step, E, ζ = rohf_manifold_linesearch(ζ, dir.vec; E, ∇E, maxstep,
                                              linesearch_type)

        # Update "info" with the new ROHF point and related quantities
        ∇E = grad_E_MO_metric(ζ.Φ, ζ)
        ∇E_prev = info.∇E; E_prev = info.E
        residual = norm(info.∇E)
        (residual<tol) && (converged=true)

        info = merge(info, (; ζ=ζ, E=E, E_prev=E_prev, ∇E=∇E, ∇E_prev=∇E_prev,
                            residual=residual, n_iter=n_iter, step=step,
                            converged=converged))
        prompt.prompt(info)

        # Choose next dir according to the solver and update info with new dir
        dir, info = solver.next_dir(info)
    end
    # Go back to non-orthonormal AO convention
    deorthonormalize_state!(ζ)
    info = merge(info, (;ζ=ζ))

    (info.converged) ? println("CONVERGED") : println("----Maximum iteration reached")

    prompt.clean(info)
end
