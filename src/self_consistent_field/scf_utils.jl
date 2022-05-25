"""
Assemble the ROHF effective Hamiltonian, that is the matrix:
+---------+---------+--------+
|  R_dd   | Fd - Fs |   Fd   |
+---------+---------+--------+
| Fd - Fs |  R_ss   |   Fs   |
+---------+---------+--------+
|   Fd    |   Fs    |  R_vv  |
+---------+---------+--------+
where R_tt = 2A_tt*(Fs)^tt + 2B_tt*(Fd-Fs)^tt
"""
function assemble_H_eff(A_tt, B_tt, Pd, Ps, Fd, Fs)
    Pv = I - (Pd + Ps)
    A_dd, A_ss, A_vv = A_tt; B_dd, B_ss, B_vv = B_tt;
    Fd_minus_s = Fd .- Fs

    # Extra diagonal terms
    H_u = Pd*(2*Fd_minus_s)*Ps + Pd*Fd*Pv + Ps*2*Fs*Pv
    # Diagonal terms
    R_dd = Pd*( A_dd*Fs + B_dd*(Fd_minus_s) )*Pd
    R_ss = Ps*( A_ss*Fs + B_ss*(Fd_minus_s) )*Ps
    R_vv = Pv*( A_vv*Fs + B_vv*(Fd_minus_s) )*Pv
    # Return H_eff
    (R_dd .+ R_ss .+ R_vv) .+ H_u .+ H_u'
end
function H_eff_coeffs(name, mol::PyObject)
    S = mol.spin
    coeffs = Dict(
        "Roothan"              => [[-0.5,0.5,1.5], [1.5,0.5,-0.5]],
        "McWeeny_Diercksen"    => [[1//3, 1//3, 2//3], [2//3, 1//3, 1//3]],
        "Davidson"             => [[1//2, 1., 1.], [1//2, 0., 0.]],
        "Guest_Saunders"       => [[1//2, 1//2, 1//2], [1//2, 1//2, 1//2]],
        "Binckey_Pople_Dobosh" => [[1//2, 1., 0.], [1//2, 0., 1.]],
        "Faegri_Manne"         => [[1//2, 1., 1//2], [1//2, 0., 1//2]],
        "Euler"                => [[1//2, 1//2, 1//2], [1//2, 0., 1//2]],
        "Canonical_1"          => [[(2*S+1.)/(2*S), 1., 1.], [-1/(2*S), 0., 0.]],
        "Canonical_2"          => [[0., 0., -1/(2*S)],  [1., 1., (2*S+1.)/(2*S)]]
    )
    coeffs[name]
end

"""
Classic self consistent field step.
1) Assemble effective hamiltonian and diagonalize.
2) Choose the first Nd+Ns MOs among eigenvectors according to the Aufbau principle.
"""
function SCF_step(Pd, Ps, Fd, Fs, mol, mo_numbers;
                  H_eff = "Roothaan",
                  )
    Nb, Nd, Ns = mo_numbers
    # Compute diagonal terms
    H_eff = assemble_H_eff(H_eff_coeffs(H_eff, mol)..., Pd, Ps, Fd, Fs)
    λ, Φ_next = eigen(Symmetric(H_eff))

    # Check Aufbau
    (λ[Nd] ≥ λ[Nd+1]) && @warn("Warning: no aufbau between ds")
    (λ[Ns] ≥ λ[Ns+1]) && @warn("Warning: no aufbau between sv")
    # Return next MOs and associated densities (Pd_next, Ps_next)
    Φ_next, densities(Φ_next, mo_numbers)
end