using PyCall; using LinearAlgebra; using DelimitedFiles

pyscf = pyimport("pyscf")

basis = "6-31g"
list_E = []

oxygen = pyscf.M(atom = "O 0.0 0.0 0.0",
    basis = "$basis", # A modifier
    symmetry = true,
    unit="bohr",
    spin=2,
    charge=0,
)

ζ = ROHFState(oxygen);
