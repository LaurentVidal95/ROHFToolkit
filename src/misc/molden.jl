function generate_molden(ζ::ROHFState, filename::String)
    pyscf.tools.molden.from_mo(ζ.Σ.mol, filename, ζ.Φ)
    nothing
end
