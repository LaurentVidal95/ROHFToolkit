Ti₂O₄(basis::String; symmetry=false) =
    ROHFToolkit.pyscf.M(
    atom = "Ti 1.730 0.000 0.000;
            Ti -1.730 0.000 0.000;
            O 0.000 1.224 0.000;
            O 0.000 -1.224 0.000;
            O 3.850 0.000 0.000;
            O -3.850 0.000 0.000;",
    basis = "$basis", # A modifier
    symmetry = symmetry,
    unit="angstrom",
    spin=2,
    charge=0,
)
