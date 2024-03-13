Fe2(basis::String; symmetry=false) =
    ROHFToolkit.pyscf.M(atom = "Fe 0.0 0.0 0.0",
                        basis = "$basis", # A modifier
                        symmetry = symmetry,
                        unit="bohr",
                        spin=4,
                        charge=2,
                        )

Fe3(basis::String; symmetry=false) =
    ROHFToolkit.pyscf.M(atom = "Fe 0.0 0.0 0.0",
                        basis = "$basis", # A modifier
                        symmetry = symmetry,
                        unit="bohr",
                        spin=5,
                        charge=3,
                        )
