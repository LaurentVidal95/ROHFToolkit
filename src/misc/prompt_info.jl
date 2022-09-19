# Add show angle of current direction w.r. to -∇E
function default_prompt(; show_dir_angle=false)
    function prompt(info)
        if info.n_iter == 0
            println("ROHF energy minimization method: $(info.solver.name)")
            println("Convergence threshold (projected gradient norm): $(info.tol)")

            header = ["Iter", "Energy","log10(ΔE)", "log10(||Π∇E||)"]
            println("-"^58)
            println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
            println("-"^58)

            info_out = [info.n_iter, info.E, " "^16, " "^16]
            println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        else
            log_ΔE = log(10, abs(info.E  - info.E_prev))
            info_out = [info.n_iter, info.E, log_ΔE, log10(info.residual)]
            println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))

            flush(stdout)
        end
    end
    # Select in "info" only the argmunents returned by the minimization procedure.
    clean(info) = (;energy=info.E, ζ=info.ζ, n_iter=info.n_iter,
                    residual=norm(info.∇E), solver=info.solver.name)
    (; prompt, clean)
end
