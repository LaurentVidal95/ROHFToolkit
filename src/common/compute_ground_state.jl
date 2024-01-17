@doc raw"""
    compute_ground_state(ζ::ROHFState; solver=ConjugateGradient, solver_kwargs...)

Wraps all ground state computation routines in a single routine.
    - ζ: Initial point of the optimization
    - solver: optimization method which can be any of the method in the
    OptimKit library(SteepestDescent, ConjugateGradient (default), LBFGS),
    or any scf method in `src/self_consistent_field/scf_solvers.jl`
    (scf, hybrid_scf).
"""
function compute_ground_state(ζ::ROHFState; solver=ConjugateGradient,
                              CASSCF=false,
                              CFOUR_ex="xcasscf",
                              solver_kwargs...)
    # Direct minimization
    if (solver ∈ (GradientDescent, ConjugateGradient, LBFGS))
        fg = CASSCF ? ζ->CASSCF_energy_and_gradient(ζ; CFOUR_ex) : energy_and_gradient
        return direct_minimization_OptimKit(ζ; solver, fg, solver_kwargs...)
    # Self consistent field
    else
        return scf_method(ζ; solver, solver_kwargs...)
    end
    error("Solver not handled")
end
