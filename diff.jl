diff --git a/src/ROHFToolkit.jl b/src/ROHFToolkit.jl
index c418c3b..cb98a75 100644
--- a/src/ROHFToolkit.jl
@@ -41,10 +41,10 @@ include("direct_minimization/preconditioning_AMO.jl")
 include("direct_minimization/OptimKit_solve.jl")
 
 export GradientDescentManual, ConjugateGradientManual, LBFGSManual
-include("direct_minimization/solve_manual/AMO_linesearch.jl")
-include("direct_minimization/solve_manual/direct_min_solvers.jl")
-include("direct_minimization/solve_manual/main_direct_minimization.jl")
-include("direct_minimization/solve_manual/prompt_info.jl")
 
 #### Self consistent field
 export scf
diff --git a/src/common/State.jl b/src/common/State.jl
index f5d6706..c8411a2 100644
--- a/src/common/State.jl
@@ -128,6 +128,8 @@ norm(X::TangentVector) = norm(X.vec)
 
 @doc raw"""
     reset_state!(ζ::State; guess=:minao)
 """
 function reset_state!(ζ::State; guess=:minao, virtuals=true)
     Φ_init = init_guess(ζ.Σ, guess; virtuals)
diff --git a/src/compute_ground_state.jl b/src/compute_ground_state.jl
index 8cbbebc..9247566 100644
--- a/src/compute_ground_state.jl
@@ -34,7 +34,7 @@ function compute_ground_state(ζ::State;
         LINESEARCHES_LOADED = (:LineSearches ∈ names(Main, imported=true))
         (!LINESEARCHES_LOADED) && error("You need to import LineSearches and assign `linesearch` "*
                                         "before launching manual direct minimization")
-        return direct_minimization_manual(ζ; solver, solver_kwargs...)
 
     # Self consistent field
     else
diff --git a/src/direct_minimization/OptimKit_solve.jl b/src/direct_minimization/OptimKit_solve.jl
index 79efc25..92e1262 100644
--- a/src/direct_minimization/OptimKit_solve.jl
@@ -41,11 +41,11 @@ function direct_minimization_OptimKit(ζ::State;
     # Optimization via OptimKit
     ζ0, E0, ∇E0, _ = optimize(fg, ζ,
                               solver(; gradtol=tol, maxiter, verbosity=0, kwargs...);
-                              optimkit_kwargs(;retraction_type=retraction,
-                                              transport_type=transport,
-                                              preconditioned,
-                                              preconditioning_trigger,
-                                              verbose)...
                               )
 
     # orthonormal AO -> non-orthonormal AO convention
@@ -63,7 +63,7 @@ Wraps all the tools needed for Riemannian optimization (retraction, projection,
 inner product, etc..) in a format readable by OptimKit.
 See `src/common/MO_manifold_tools.jl`
 """
-function optimkit_kwargs(; retraction_type=:exp,
                       transport_type=:exp,
                       preconditioned=true,
                       preconditioning_trigger=10^(-0.5),
@@ -85,7 +85,8 @@ function optimkit_kwargs(; retraction_type=:exp,
 end
 
 function precondition(ζ::State, η; trigger=10^(-0.5))
-    ∇E_prec_vec = preconditioned_gradient_AMO(ζ; trigger)
     TangentVector(∇E_prec_vec, ζ)
 end
 
@@ -96,15 +97,6 @@ function retract(ζ::State{T}, η::TangentVector{T}, α; type=:exp) where {T<:Re
     Rη, τη
 end
 
-function transport!(η1::TangentVector{T}, ζ::State{T},
-                    η2::TangentVector{T}, α::T, Rη2::State{T};
-                    type=:exp) where {T<:Real}
-    # Test colinearity
-    angle(X::Matrix,Y::Matrix) = tr(X'Y) / (norm(X)*norm(Y))
-    collinear = norm(abs(angle(η1.vec, η2.vec)) - 1) < 1e-8
-    transport_AMO(η1, ζ, η2, α, Rη2; type, collinear)
-end
-
 inner(ζ::State, η1::TangentVector, η2::TangentVector) = tr(η1'η2)
 function scale!(η::TangentVector{T}, α) where {T<:Real}
     TangentVector(α*η, η.base)
@@ -114,6 +106,15 @@ function add!(η1::TangentVector{T}, η2::TangentVector{T},
     TangentVector(η1 + α*η2, η1.base)
 end
 
 """
 Equivalent of the prompt routine but with OptimKit conventions
 """
diff --git a/src/direct_minimization/solve_manual/AMO_linesearch.jl b/src/direct_minimization/manual_direct_minimization/AMO_linesearch.jl
similarity index 81%
rename from src/direct_minimization/solve_manual/AMO_linesearch.jl
rename to src/direct_minimization/manual_direct_minimization/AMO_linesearch.jl
index 896fbfd..deab83c 100644
--- a/src/direct_minimization/solve_manual/AMO_linesearch.jl
@@ -10,35 +10,29 @@ The ``linesearch_type`` can be any of the linesearch algorithms in the LineSearc
 function AMO_linesearch(ζ::State{T}, p::TangentVector{T};
                                E, ∇E, linesearch_type,
                                maxstep = one(Float64),
-                               retraction=:exp,
-                               transport=:exp
                                ) where {T<:Real}
     # All linesearch routines are performed in orthonormal AOs convention
     @assert(ζ.isortho)
-
-    # DEBUG
-    test_1 = test_tangent(p)
-    test_2 = tr(∇E'p)/(norm(∇E)*norm(p))
-    (test_1 > 1e-8) && (@show test_1)
-    (test_2 > -1e-2) && (@show test_2)
-    
     # LineSearches.jl objects
     function f(step)
-        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction)
         energy(ζ_next)
     end
 
     function df(step)
-        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction)
-        τ_p = transport_AMO(p, ζ, p, step, ζ_next; type=transport, collinear=true)
         ∇E_next = AMO_gradient(ζ_next)
         tr(∇E_next'τ_p)
     end
 
     function fdf(step)
-        ζ_next = retract_AMO(ζ, TangentVector(step .* p.vec, ζ); type=retraction)
         E_next, ∇E_next =  energy_and_riemannian_gradient(ζ_next)
-        τ_p = transport_AMO(p, ζ, p, step, ζ_next; type=transport, collinear=true)
         E_next, tr(∇E_next'τ_p)
     end
 
@@ -48,7 +42,7 @@ function AMO_linesearch(ζ::State{T}, p::TangentVector{T};
 
     # Actualize ζ and energy
     
-    ζ_next = retract_AMO(ζ, TangentVector(α .* p.vec, ζ); type=retraction)
     ζ_next.energy = E_next
     @assert (test_MOs(ζ_next) < 1e-8)
 
diff --git a/src/direct_minimization/solve_manual/direct_min_solvers.jl b/src/direct_minimization/manual_direct_minimization/direct_min_solvers.jl
similarity index 76%
rename from src/direct_minimization/solve_manual/direct_min_solvers.jl
rename to src/direct_minimization/manual_direct_minimization/direct_min_solvers.jl
index 16a163c..64367b0 100644
--- a/src/direct_minimization/solve_manual/direct_min_solvers.jl
@@ -11,22 +11,23 @@ struct GradientDescentManual <: Solver
     name           ::String
     prefix         ::String
     preconditioned ::Bool
 end
-function GradientDescentManual(; preconditioned=true)
     name = preconditioned ? "Preconditioned Steepest Descent" : "Steepest Descent"
     prefix = preconditioned ? "prec_SD" : "SD"
-    GradientDescentManual(name, prefix, preconditioned)
 end
 
-function next_dir(S::GradientDescentManual, info; precondition, kwargs...)
-    grad_vec = S.preconditioned ? precondition(info.ζ) : info.∇E
     dir = TangentVector(-grad_vec, info.ζ)
     dir, merge(info, (; dir))
 end
 
 
 @doc raw"""
-    OLD: ConjugateGradientManual(; preconditioned=true, flavor="Fletcher-Reeves")
 
 (Preconditioned) conjugate gradient algorithm on the MO manifold.
 The ``cg_type`` for now is useless but will serve to launch other
@@ -36,41 +37,41 @@ struct ConjugateGradientManual <: Solver
     name           ::String
     prefix         ::String
     preconditioned ::Bool
-    flavor        ::Symbol
-end
-function ConjugateGradientManual(; preconditioned=true, flavor=:Fletcher_Reeves)
-    @assert flavor ∈ (:Fletcher_Reeves, :Polack_Ribiere)
     name = preconditioned ? "Preconditioned Conjugate Gradient" : "Conjugate Gradient"
     prefix = preconditioned ? "prec_CG" : "CG"
-    ConjugateGradientManual(name, prefix, preconditioned, flavor)
 end
 
-function next_dir(S::ConjugateGradientManual, info; precondition, transport,
-                  kwargs...)
     ζ = info.ζ
     ∇E = info.∇E;  ∇E_prev = info.∇E_prev;
     dir = info.dir
-    current_grad = S.preconditioned ? precondition(ζ) : ∇E
 
     # Transport previous dir and gradient on current point ζ
-    τ_dir_prev = transport_AMO(dir, dir.base, dir, 1., ζ; type=transport, collinear=true)
 
     # Assemble CG dir with Fletcher-Reeves or Polack-Ribiere coefficient
-    β = zero(ζ.energy)
-    begin
-        cg_factor = zero(β)
-        if S.flavor==:Fletcher_Reeves
-            τ_grad_prev = transport_AMO(∇E_prev, ∇E_prev.base, dir, 1., ζ; type=transport, collinear=false)
-            cg_factor = tr(∇E'τ_grad_prev)
         end
-        β = (tr(∇E'current_grad) - cg_factor) / norm(info.∇E_prev)^2 # DEBUG: wrong use of preconditioning ?
-        # Restart if not a descent direction
-        dir = TangentVector(project_tangent_AMO(ζ, -current_grad + β*τ_dir_prev), ζ)
-        (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2) && (β = zero(β))        
-        # Restart if β is negative
-        # β = (β > 0) ? β : zero(Float64)
     end
-    iszero(β) && (@warn "Restart")
     dir = TangentVector(project_tangent_AMO(ζ, -current_grad + β*τ_dir_prev), ζ)
     dir, merge(info, (; dir))
 end
@@ -83,14 +84,15 @@ struct LBFGSManual <: Solver
     prefix         ::String
     preconditioned ::Bool
     depth          ::Int
 end
-function LBFGSManual(;depth=8, preconditioned=:false)
     name = preconditioned ? "Preconditioned LBFGS" : "LBFGS"
     prefix = preconditioned ? "prec_LBFGS" : "LBFGS"
-    LBFGSManual(name, prefix, preconditioned, depth)
 end
 
-function next_dir(S::LBFGSManual, info; preconditioner, transport, kwargs...)
     # Extract data
     B = info.B
     x_prev = info.dir.base;  ∇E_prev = info.∇E_prev
@@ -124,15 +126,8 @@ function next_dir(S::LBFGSManual, info; preconditioner, transport, kwargs...)
     push!(B, (s,y,ρ))
 
     # Compute next dir
-    dir_vec = -B(∇E; B₀=default_LBFGS_init)
     dir = TangentVector(-B(∇E).vec, x_new)
-
-    # Restart BFGS if dir is not a descent direction
-    if (tr(dir'∇E)/(norm(dir)*norm(∇E)) > -1e-2)
-        @warn "Restart: not a descent direction"
-        empty!(B)
-    end
-        
     dir, merge(info, (; dir, B))
 end
 
diff --git a/src/direct_minimization/solve_manual/main_direct_minimization.jl b/src/direct_minimization/manual_direct_minimization/main_direct_minimization.jl
similarity index 64%
rename from src/direct_minimization/solve_manual/main_direct_minimization.jl
rename to src/direct_minimization/manual_direct_minimization/main_direct_minimization.jl
index 59c862e..76fb992 100644
--- a/src/direct_minimization/solve_manual/main_direct_minimization.jl
@@ -1,32 +1,31 @@
 @doc raw"""
-    OLD: direct_minimization(ζ::State;  TODO)
 
 General direct minimization procedure, which decomposes as such:
     1) Choose a direction according to the method provided in the solver arg.
     2) Linesearch along direction
     3) Check convergence
 The arguments are
-    TODO
 """
 function direct_minimization_manual(ζ::State;
                                     maxiter = 500,
                                     maxstep = 2*one(Float64),
                                     tol = 1e-5,
-                                    # Choose solver and preconditioning
-                                    solver=ConjugateGradientManual,
-                                    preconditioned=true,
-                                    preconditioning_trigger=10^(-0.5),
-                                    # Type of retraction and transport
-                                    retraction=:exp,
-                                    transport=:exp,
-                                    linesearch,
-                                    # Prompt
                                     prompt=default_direct_min_prompt(),
                                     solver_kwargs...)
-
-    # Setup solver and preconditioner
-    precondition(ζ) = preconditioned_gradient_AMO(ζ; trigger=preconditioning_trigger)
-    sol = solver(; solver_kwargs...)
 
     # Linesearch.jl only handles Float64 step sisze
     (typeof(maxstep)≠Float64) && (maxstep=Float64(maxstep))
@@ -40,18 +39,18 @@ function direct_minimization_manual(ζ::State;
     n_iter          = zero(Int64)
     E, ∇E           = energy_and_riemannian_gradient(ζ)
     E_prev, ∇E_prev = E, ∇E
-    dir_vec         = sol.preconditioned ? .- precondition(ζ) : - ∇E
     dir             = TangentVector(dir_vec, ζ)
     step            = zero(Float64)
     converged       = false
     residual        = norm(∇E)
 
-    info = (; n_iter, ζ, E, E_prev, ∇E, ∇E_prev, dir, solver=sol,
-            step, converged, tol, residual)
 
     # init LBFGS solver if needed
-    if isa(sol, LBFGSManual)
-        B = LBFGSInverseHessian(sol.depth, TangentVector[],  TangentVector[], eltype(E)[])
         info = merge(info, (; B))
     end
 
@@ -63,8 +62,7 @@ function direct_minimization_manual(ζ::State;
 
         # find next point ζ on ROHF manifold
         step, E, ζ = AMO_linesearch(ζ, dir; E, ∇E, maxstep,
-                                    linesearch_type=linesearch,
-                                    retraction, transport)
 
         # Update "info" with the new ROHF point and related quantities
         ∇E = AMO_gradient(ζ)
@@ -77,7 +75,7 @@ function direct_minimization_manual(ζ::State;
         prompt.prompt(info)
 
         # Choose next dir according to the solver and update info with new dir
-        dir, info = next_dir(sol, info; precondition, transport)
     end
     # Go back to non-orthonormal AO convention
     deorthonormalize_state!(ζ)
diff --git a/src/direct_minimization/solve_manual/prompt_info.jl b/src/direct_minimization/manual_direct_minimization/prompt_info.jl
similarity index 100%
rename from src/direct_minimization/solve_manual/prompt_info.jl
rename to src/direct_minimization/manual_direct_minimization/prompt_info.jl
diff --git a/src/direct_minimization/preconditioning_AMO.jl b/src/direct_minimization/preconditioning_AMO.jl
index dbd078f..4b6dea4 100644
--- a/src/direct_minimization/preconditioning_AMO.jl
@@ -1,4 +1,4 @@
-function preconditioned_gradient_AMO(ζ::State; num_safety=1e-6, trigger=10^(-1/2))
     Fi, Fa = Fock_operators(ζ)
     H, G_vec = build_quasi_newton_system(ζ.Φ, Fi, Fa, ζ.Σ.mo_numbers;
                                          num_safety)
@@ -7,20 +7,18 @@ function preconditioned_gradient_AMO(ζ::State; num_safety=1e-6, trigger=10^(-1/
         X, Y, Z = reshape_XYZ(XYZ, Ni, Na, Ne)
         [zeros(Ni,Ni) X Y; -X' zeros(Na,Na) Z; -Y' -Z' zeros(Ne,Ne)]
     end
-
-    # Return standard gradient if to far from a minimum
-    Nb, Ni, Na = ζ.Σ.mo_numbers
-    Ne = Nb - (Ni+Na)
-    κ_grad = vec_to_κ(G_vec, Ni, Na, Ne)
-    (norm(κ_grad)>trigger) && (return (ζ.Φ*κ_grad))
-
     # Compute quasi newton direction by solving the system with BICGStab l=3.
     # BICGStab requires that L is positive definite which is not the case
     # far from a minimum. In practice a numerical hack allows to compute
     # the lowest eigenvalue of L (without diagonalizing) and we apply a level-shift.
     # Otherwise, replace by Minres if L is still non-positive definite.
     XYZ, history = bicgstabl(H, G_vec, 3, reltol=1e-14, abstol=1e-14, log=true)
-    κ = vec_to_κ(XYZ, Ni, Na, Ne)     # Convert back to matrix format
 
     # Return unpreconditioned grad if norm(κ) is too high
     # or if -prec_grad is not a descent direction
