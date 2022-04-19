using PyCall; using LinearAlgebra; using DelimitedFiles
using Pkg; Pkg.activate("/home/lvidal/Documents/CERMICS/these/rohf_papers/code/ROHFToolkit_MO"); using ROHFToolkit

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
