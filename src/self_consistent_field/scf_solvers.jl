using NLsolve

# SCF with no diis
function scf()
    (;solve=x->nothing, prefix="SCF", name="SCF without DIIS")
end

# TODO
function oda()
    nothing
end

# TODO
function g_new_diis()
    nothing
end

# TODO ADAPT WITH OUR OWN DIIS
# """
# Create a simple anderson-accelerated SCF solver. `m` specifies the number
# of steps to keep the history of.
# """
# function scf_anderson_solver(m=10; kwargs...)
#     function anderson(f, x0, max_iter; tol=1e-6)
#         T = eltype(x0)
#         x = x0

#         converged = false
#         acceleration = AndersonAcceleration(;m=m, kwargs...)
#         for n = 1:max_iter
#             residual = f(x)[1] .- x
#             converged = norm(residual) < tol
#             converged && break
#             x = acceleration(x, one(T), residual)
#         end
#         (fixpoint=x, converged=converged)
#     end
#     solve = anderson
#     (;solve, name="Self consistent field with DIIS", prefix="SCF_manual_DIIS")
# end
