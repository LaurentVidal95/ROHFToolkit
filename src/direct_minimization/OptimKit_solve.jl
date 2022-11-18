using OptimKit

"""
Wrapper around OptimKit "optimize" function.
"""
function direct_minimization_OptimKit(ζ::ROHFState;
                                      maxiter=500,
                                      tol=1e-5,
                                      solver=ConjugateGradient,
                                      preconditioned=true,
                                      verbose=true,
                                      break_symmetry=false,
                                      kwargs...)
    # non-orthonormal AO -> orthonormal AO convention
    # TODO add fix in case of high conditioning number
    (cond(ζ.Σ.overlap_matrix) > 1e6) && @warn("Conditioning of the "*
                                 "overlap: $(cond(ζ.Σ.overlap_matrix))")
    orthonormalize_state!(ζ)
    (break_symmetry) && (@warn "Broken symmetry"; ζ.Φ = ζ.Φ*rand_unitary_matrix(ζ))
    # Optimization via OptimKit
    ζ0, E0, ∇E0, _ = optimize(energy_and_gradient, ζ,
                              solver(; gradtol=tol, maxiter, verbosity=0);
                              optim_kwargs(;preconditioned, verbose)...)
    # orthonormal AO -> non-orthonormal AO convention
    deorthonormalize_state!(ζ0)
    (norm(∇E0)>tol) && (@warn "Not converged")
    @info "Final energy: $(E0) Ha"

    (;ζ=ζ0, energy=E0, residual=norm(∇E0))
end

"""
All manifold routines in a format readable by OptimKit.
"""
function optim_kwargs(;preconditioned=true, verbose=true)
    kwargs = (; retract, inner, transport!, scale!, add!)
    (verbose) && (kwargs=merge(kwargs, (; finalize!)))
    (preconditioned) && (kwargs=merge(kwargs, (;precondition)))
    kwargs
end

function precondition(ζ::ROHFState, η) where {T<:Real}
    prec_grad = ROHFTangentVector(preconditioned_gradient_MO_metric(ζ), ζ)
    #return gradient if not a descent direction. Avoid errors in L-BFGS far from minimum
    if (tr(prec_grad'η)/(norm(prec_grad)*norm(η))) ≤ 1e-5
        @warn "No preconditioning"
        return η
    end
    prec_grad
end

function retract(ζ::ROHFState{T}, η::ROHFTangentVector{T}, α) where {T<:Real}
    @assert(η.base.Φ == ζ.Φ) # check that ζ is the base of η
    Rη = ROHFState(ζ, retract(ζ, α*η, ζ.Φ))
    τη = transport_vec_along_himself(η, α, Rη)
    Rη, τη
end

inner(ζ::ROHFState, η1::ROHFTangentVector, η2::ROHFTangentVector) = tr(η1'η2)
@inline function scale!(η::ROHFTangentVector{T}, α) where {T<:Real}
    ROHFTangentVector(α*η, η.base)
end
@inline function add!(η1::ROHFTangentVector{T}, η2::ROHFTangentVector{T},
                      α::T2) where {T<:Real, T2<:Real}
    ROHFTangentVector(η1 + α*η2, η1.base)
end

function transport!(η1::ROHFTangentVector{T}, ζ::ROHFState{T},
                    η2::ROHFTangentVector{T}, α::T, Rη2::ROHFState{T}) where {T<:Real}
    # η1 = η2 case
    (η1.vec==η2.vec) && (return transport_vec_along_himself(η1, α, Rη2))
    # Otherwise transport with projection1;5D
    τη1_vec = project_tangent(ζ, Rη2.Φ, η1.vec)
    ROHFTangentVector(τη1_vec, Rη2)
end

"""
Equivalent of the prompt routine but with OptimKit conventions
"""
function finalize!(ζ, E, ∇E, n_iter)
    if n_iter == 1
        println("ROHF direct energy minimization")
        println("Initial guess: $(ζ.guess)")
        header = ["Iter", "Energy","log10(ΔE)", "log10(||Π∇E||)"]
        println("-"^58)
        println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
        println("-"^58)

        info_out = [n_iter-1, ζ.energy, " "^16, " "^16]
        println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        flush(stdout)
    end
    # Print current iter infos
    log_ΔE = log(10, abs(E  - ζ.energy))
    residual = norm(∇E)
    info_out = [n_iter, E, log_ΔE, log10(residual)]
    println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))
    flush(stdout)

    # Actualize energy
    energy!(ζ)
    # Update history
    ζ.history = vcat(ζ.history, reshape(info_out, 1, 4))

    # Return entry to match OptimKit.jl conventions
    ζ, E, ∇E
end
