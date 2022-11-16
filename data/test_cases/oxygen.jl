using ROHFToolkit

basis = "6-31g"
list_E = []

oxygen = ROHFToolkit.pyscf.M(atom = "O 0.0 0.0 0.0",
    basis = "$basis", # A modifier
    symmetry = true,
    unit="bohr",
    spin=2,
    charge=0,
)

ζ = ROHFState(oxygen; guess=:hcore);
# res = ROHF_ground_state(ζ)
