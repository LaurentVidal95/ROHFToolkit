# Add show angle of current direction w.r. to -∇E
function default_direct_min_prompt(; show_dir_angle=false)
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
            residual = log(10, norm(info.∇E))
            info_out = [info.n_iter, info.E, log_ΔE, residual]
            println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))

            flush(stdout)
        end
    end
    # Select in "info" only the argmunents returned by the minimization procedure.
    clean(info) = (;energy=info.E, ζ=info.ζ, n_iter=info.n_iter,
                    residual=norm(info.∇E), solver=info.solver.name)
    (; prompt, clean)
end

function default_scf_prompt()
    function prompt(info)
        if info.n_iter == 0
            println("ROHF energy minimization method: ROHF SCF with"*
                    " $(info.effective_hamiltonian) coefficiens")
            println("Convergence threshold: $(info.tol)")
            
            header = ["Iter", "Energy", "log10(ΔE)", "log10(||Residual||)"]
            println("-"^58)
            println(@sprintf("%-5s  %-16s  %-16s  %-16s", header...))
            println("-"^58)
            
            info_out = [info.n_iter, info.E, " "^16, " "^16]
            println(@sprintf("%5i %16.12f %16s %16s", info_out...))
        else
            log_ΔE = log(10, abs(info.E  - info.E_prev))
            residual = log(10, norm(info.residual))
            info_out = [info.n_iter, info.E, log_ΔE, residual]
            println(@sprintf("%5i %16.12f %16.12f %16.12f", info_out...))
            
            flush(stdout)
        end 
    end
    # Select in "info" only the argmunents returned by the minimization procedure.
    clean(info) = (;energy=info.E, ζ=info.ζ, n_iter=info.n_iter,
                   residual=norm(info.∇E))
    (; prompt, clean)
end
