# TODO: add g_new SCF
using NLsolve

function scf_diis(; m=10, method=:anderson, kwargs...)    
    function fp_nl_solver(f, x0, max_iter; tol=1e-6)
        res = nlsolve(x -> f(x) - x, x0; method=method, m=m, xtol=tol,
                      ftol=0.0, show_trace=false, iterations=max_iter, kwargs...)
        (fixpoint=res.zero, converged=converged(res))
    end
    solve = fp_nl_solver
    (;solve, name="Self consistent field with DIIS", prefix="SCF_DIIS")
end
