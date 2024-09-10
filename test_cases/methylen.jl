# START FROM A CORE GUESS FOR PROPER SYMMETRY GROUND STATE
methylen(basis::String; symmetry=false) =
    ROHFToolkit.pyscf.M(
    atom = "C        0.00000000      -0.00000000       0.06143027;
            H        0.00000000      -0.98921656      -0.36571986;
            H        0.00000000       0.98921656      -0.36571986",
    basis = "$basis", # A modifier
    symmetry = symmetry,
    unit="angstrom",
    spin=2,
    charge=0,
)
