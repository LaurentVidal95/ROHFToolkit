# START FROM A CORE GUESS FOR PROPER SYMMETRY GROUND STATE
hydroxyl(basis::String; symmetry=false) =
    ROHFToolkit.pyscf.M(
    atom = "H 0.000 0.000 0.000;
            O 0.000 0.000 1.01",
    basis = "$basis", # A modifier
    symmetry = symmetry,
    unit="angstrom",
    spin=1,
    charge=0,
)
