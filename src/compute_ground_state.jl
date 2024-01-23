@doc raw"""
    compute_ground_state(ζ::State; solver=ConjugateGradient, solver_kwargs...)

Wraps all ground state computation routines in a single routine.
    - ζ: Initial point of the optimization
    - solver: optimization method which can be any of the method in the
    OptimKit library(SteepestDescent, ConjugateGradient (default), LBFGS),
    or any scf method in `src/self_consistent_field/scf_solvers.jl`
    (scf, hybrid_scf).
"""
function compute_ground_state(ζ::State;
                              # Interface with CFOUR CASSCF code
                              CASSCF=false,
                              CFOUR_ex="xcasscf",
                              CASSCF_verbose=false,
                              # Solver and related args
                              solver=ConjugateGradient,
                              solver_kwargs...)
    # Direct minimization
    if (solver ∈ (GradientDescent, ConjugateGradient, LBFGS))
        CASSCF_kwargs = (; CFOUR_ex, verbose=CASSCF_verbose)
        fg = begin
            if CASSCF
                ζ->CASSCF_energy_and_gradient(ζ; CASSCF_kwargs...)
            else
                energy_and_riemannian_gradient
            end
        end
        return direct_minimization_OptimKit(ζ; solver, fg, solver_kwargs...)
    # Self consistent field
    else
        return scf_method(ζ; solver, solver_kwargs...)
    end
    error("Solver not handled")
end
