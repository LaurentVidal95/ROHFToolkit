@doc raw"""
    direct_minimization(ζ::State;  TODO)

General direct minimization procedure, which decomposes as such:
    1) Choose a direction according to the method provided in the solver arg.
    2) Linesearch along direction
    3) Check convergence
The arguments are
    TODO
"""
function direct_minimization(ζ::State;
                             maxiter = 500,
                             maxstep = 2*one(Float64),
                             tol = 1e-5,
                             #  Type of retraction and transport
                             retraction_type=:exp,
                             transport_type=:exp,
                             # Prompt
                             prompt=default_direct_min_prompt(),
                             # Choose solver and preconditioning
                             solver=ConjugateGradient,
                             preconditioned=true,
                             preconditioner=default_preconditioner,
                             linesearch,
                             # Choose between ROHF or CASSCF functionals
                             f=ROHF_energy,
                             g=ROHF_gradient,
                             fg=ROHF_energy_and_gradient,
                             # CI tol parameters (only for CASSCF)
                             tolmin=1e-2,
                             tolmax=1e-10,
                             solver_kwargs...)

    # Setup solver and preconditioner
    sol = solver(; preconditioned, solver_kwargs...)
    !(preconditioned) && (preconditioner=∇E->∇E)

    # Linesearch.jl only handles Float64 step sisze
    (typeof(maxstep)≠Float64) && (maxstep=Float64(maxstep))

    # non-orthonormal AO -> orthonormal AO convention
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                     "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)

    # Populate info with initial data
    n_iter          = zero(Int64)
    tol_ci          = 0.1*tolmin
    E, ∇E           = fg(ζ; tol_ci)    
    E_prev, ∇E_prev = copy(E), deepcopy(∇E)
    P∇E             = preconditioner(∇E)
    P∇E_prev        = deepcopy(P∇E)
    dir             = TangentVector(-P∇E_prev, ζ)
    step            = zero(Float64)
    converged       = false
    residual        = norm(∇E)

    info = (; n_iter, ζ, E, E_prev, ∇E, ∇E_prev, P∇E, P∇E_prev, dir,
            solver=sol, step, converged, tol, residual)

    # init LBFGS solver if needed
    if isa(sol, LBFGS)
        B = LBFGSInverseHessian(sol.depth, TangentVector[],  TangentVector[],
                                TangentVector[], eltype(E)[])
        info = merge(info, (; B))
    end

    # Display header and initial data
    prompt.prompt(info)

    while (!(info.converged) && (n_iter < maxiter))
        n_iter += 1

        # find next point ζ on ROHF manifold
        step, E, ζ = AMO_linesearch(ζ, dir, f, g, X->fg(X; tol_ci); E, ∇E, maxstep,
                                    linesearch_type=linesearch,
                                    retraction_type,
                                    transport_type
                                    )
        # Update "info" with the new ROHF point and related quantities
        # TODO: add better handling of CASSCF tol_ci
        tol_ci = max(min(tolmin, 1e-3*norm(info.∇E)), tolmax)
        E, ∇E = fg(ζ; tol_ci)
        E_prev = info.E; ∇E_prev = info.∇E;
        P∇E = preconditioner(∇E); P∇E_prev=info.P∇E
        residual = norm(info.∇E)
        (residual<tol) && (converged=true)

        info = merge(info, (; ζ, E, E_prev, ∇E, ∇E_prev, P∇E, P∇E_prev, residual,
                            n_iter, step, converged))
        prompt.prompt(info)

        # Choose next dir according to the solver and update info with new dir
        dir, info = next_dir(sol, info; preconditioner, transport_type)
    end
    # Go back to non-orthonormal AO convention
    deorthonormalize_state!(ζ)
    info = merge(info, (;ζ))

    (info.converged) ? (@info "Final energy: $(ζ.energy) Ha") :
        println("----Maximum iteration reached")

    # DUMMY CFOUR INTERFACE: remove CASSCF LBFGS files
    isfile("casscf_diag.txt") && rm("casscf_diag.txt")
    isfile("RESTART_LBFGS") && rm("RESTART_LBFGS")

    prompt.clean(info)
end
