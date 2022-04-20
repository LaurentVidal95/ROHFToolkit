# Add show angle of current direction w.r. to -∇E
function default_prompt(; show_dir_angle=false)
    function prompt(info)
        if info.n_iter == 0
            println("ROHF energy minimization method: $(info.solver.name)")
            println("Convergence threshold : $(info.cv_threshold)")

            header = ["Iter", "Energy","log10(ΔE)", "log10(||Π∇E||)"]
            println("-"^57)
            println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
            println("-"^57)

            info_out = [info.n_iter, info.E, " "^16, " "^16]
            println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        else
            log_ΔE = log(10, abs(info.E  - info.E_prev))
            residual = log(10, norm(info.∇E))
            info_out = [info.n_iter, info.E, log_ΔE, residual]
            println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))

            flush(stdout)
        end
    end
end

"""
Info keeping only argmunents returned by minimization procedure.
"""
clean_info(info) = (;E=info.E, ζ=info.ζ, n_iter=info.n_iter,
                    residual=norm(info.∇E), solver=info.solver.name)
