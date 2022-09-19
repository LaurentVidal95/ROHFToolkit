using PyCall;
pyscf = pyimport("pyscf")

using ROHFToolkit


basis = "6-31g"
list_E = []

pyridine_FE2 = pyscf.M(
    atom = "C -2.1853 -1.50067 0.662721;
    N -2.7965 -2.55019 1.26604;
    C -2.4724 -1.10719 -0.670485;
    C -3.42828 -1.85803 -1.39688;
    C -4.05578 -2.9613 -0.774398;
    C -3.70334 -3.26223 0.561698;
    H -4.1526 -4.10453 1.0798;
    H -4.77004 -3.58087 -1.30627;
    H -3.66073 -1.59882 -2.42527;
    H -1.99868 -0.215119 -1.09376;
    H -1.4973 -0.929194 1.27798;
    Fe -4.28268 -1.18964 0.365066",
    basis = "$basis", # A modifier
    symmetry = true,
    unit="angstrom",
    spin=4,
    charge=2,
)

ζ = ROHFState(pyridine_FE2);
