Julia code to compare SCF and direct minimization algorithms for the high-spin
ROHF model. This code is built as an interface between the [PySCF](https://pyscf.org/)
python library, and the [OptimKit.jl](https://github.com/Jutho/OptimKit.jl)
Riemannian optimization package in Julia. BEWARE: the OptimKit library
is not maintained anymore, and the code might break with Julia updates.

# Requirements:
Julia 1.8 and above (tested up until Julia 1.10.0-rc2).

# Installing all dependancies
Open a Julia shell with `julia --project` in your local copy of this repository and call
```
using Pkg; Pkg.instantiate(".")
``` 
to install all the needed dependencies. When everything is installed, you should be
able to import the ROHFToolkit package with
```
using ROHFToolkit
```

# Interface with PySCF
In order to guarantee that the code runs at optimal speed, it is better
to use the python environment created by [Conda.jl](https://github.com/JuliaPy/Conda.jl).
In a fresh julia shell, install the Conda.jl with
```
using Pkg; Pkg.add("Conda")
```
Then install the pyscf library in Conda environment with
```
using Conda; Conda.add("pyscf")
```
Now open a shell in the ROHFToolkit project as in the previous section.
Set up the PyCall python environment with
```
ENV["PYTHON"] = "/home/username/.julia/conda/3/bin/python_with_pyscf"
```
and call
```
using Pkg; Pkg.build("PyCall")
```
The code should be running fine, and fast.
# Launch a ROHF computation

Let us compute the ground state of oxygen in a 6-31G basis, from the core
Hamiltonian guess, using a preconditioned Riemannian conjugate gradient
algorithm. The oxygen molecule as a PySCF object has been defined in the 
``test_cases`` directory:
```
using ROHFToolkit

include("test_cases/oxygen.jl")
x_init = ROHFState(oxygen("6-31G"), guess=:hcore)

# Launch minimization. (Note that these kwargs are the default one)
output = compute_ground_state(x_init; solver=ConjugateGradient, preconditioned=true)

# Extract final state and energy
x_min = output.final_state
e_min = output.energy
```
We can then generate a `molden` file for visualization with
```
generate_molden(x_min, "oxygen_min_orbitals.mol")
```

# Contact
This is research code, not user-friendly, actively maintained, extremely robust
or optimized. If you have questions contact me at:
Laurent(dot)vidal(at)enpc(dot)fr
