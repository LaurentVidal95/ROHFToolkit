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
                              tolmin=1e-2,
                              tolmax=1e-10,
                              # Solver and related args
                              solver=ConjugateGradient,
                              # linesearch=nothing, # for manual only
                              solver_kwargs...)
    # Choose between CASSCF and ROHF
    CASSCF_kwargs = (; CFOUR_ex, verbose=CASSCF_verbose)
    f, g, fg = begin
        if CASSCF
            f(x::State) = CASSCF_energy(x; CASSCF_kwargs...)
            g(x::State) = CASSCF_gradient(x; CASSCF_kwargs...)
            fg(x::State) = CASSCF_energy_and_gradient(x; CASSCF_kwargs...)
            f, g, fg
        else
            ROHF_energy, ROHF_gradient, ROHF_energy_and_gradient
        end
    end

    # Direct minimization
    if (solver ∈ (GradientDescent, ConjugateGradient, LBFGS))
        return direct_minimization_OptimKit(ζ; solver, fg, solver_kwargs...)

    # Manual direct minimization
    elseif (solver ∈ (GradientDescentManual, ConjugateGradientManual, LBFGSManual))
        # Manual solvers need external LineSearches.jl library
        LINESEARCHES_LOADED = (:LineSearches ∈ names(Main, imported=true))
        (!LINESEARCHES_LOADED) && error("You need to import LineSearches and assign `linesearch` "*
                                        "before launching manual direct minimization")
        return direct_minimization_manual(ζ; f, g, fg, solver, tolmin, tolmax,
                                          solver_kwargs...)

    # Self consistent field (only works for ROHF)
    else
        @assert !CASSCF "SCF method only for ROHF"
        return scf_method(ζ; solver, solver_kwargs...)
    end

    error("Solver not handled")
end
