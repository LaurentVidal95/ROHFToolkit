# Add show angle of current direction w.r. to -∇E
function default_direct_min_prompt(; show_dir_angle=false)
    function prompt(info)
        if info.n_iter == 0
            ζ = info.ζ
            res = log(10, norm(info.∇E))
            ζ.history = reshape([0, info.E, NaN, res, NaN], 1, 5)
            info = merge(info, (;ζ))

            println("ROHF energy minimization method: $(info.solver.name)")
            println("Convergence threshold (projected gradient norm): $(info.tol)")

            header = ["Iter", "Energy","log10(ΔE)", "log10(||Π∇E||)", "step"]
            println("-"^76)
            println(@sprintf("%-5s  %-16s  %-16s  %-16s %-16s", header...))
            println("-"^76)

            info_out = [info.n_iter, info.E, " "^16, res, " "^16]
            println(@sprintf("%5i %16.12f %16s %16.12f %16s", info_out...))
        else
            ζ = info.ζ
            log_ΔE = log(10, abs(info.E  - info.E_prev))
            residual = log(10, norm(info.∇E))
            info_out = [info.n_iter, info.E, log_ΔE, residual, info.step]

            ζ.history = vcat(ζ.history, reshape(info_out, 1, 5))
            info = merge(info, (;ζ))

            println(@sprintf("%5i %16.12f %16.12f %16.12f %16.12f", info_out...))

            flush(stdout)
        end
    end
    # Select in "info" only the argmunents returned by the minimization procedure.
    clean(info) = (;energy=info.E, final_state=info.ζ, info.n_iter,
                    residual=norm(info.∇E), solver=info.solver.name)
    (; prompt, clean)
end

function default_scf_prompt()
    function prompt(info)
        if info.n_iter == 0
            ζ = info.ζ
            res = log(10, norm(info.∇E))
            ζ.history = reshape([0, info.E, NaN, res, NaN], 1, 5)
            info = merge(info, (;ζ))

            # Small hack to print steps also
            println("ROHF energy minimization method: ROHF SCF with"*
                    " $(info.effective_hamiltonian) coefficiens")
            println("Convergence threshold: $(info.tol)")

            header = ["Iter", "Energy", "log10(ΔE)", "log10(||Residual||)"]
            println("-"^58)
            println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
            println("-"^58)

            info_out = [info.n_iter, info.E, " "^16, res]
            println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        else
            log_ΔE = log(10, abs(info.E  - info.E_prev))
            residual = log(10, norm(info.residual))
            info_out = [info.n_iter, info.E, log_ΔE, residual]
            # Actualize history
            println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))

            flush(stdout)
        end
    end
    # Select in "info" only the argmunents returned by the minimization procedure.
    clean(info) = (;energy=info.E, info.ζ, info.n_iter,
                   residual=norm(info.∇E))
    (; prompt, clean)
end
