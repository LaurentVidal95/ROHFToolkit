oxygen(basis::String; symmmetry=false) =
    ROHFToolkit.pyscf.M(atom = "O 0.0 0.0 0.0",
                        basis = "$basis", # A modifier
                        symmetry = symmetry,
                        unit="bohr",
                        spin=2,
                        charge=0,
                        )
