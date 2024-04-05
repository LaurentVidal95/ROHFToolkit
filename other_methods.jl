using PyCall

# SOSCF with pyscf
function soscf(x::State; guess="minao", verbosity=4)
    mol = x.Σ.mol
    mol.verbosity=verbosity
    mf = pyscf.scf.ROHF(mol)
    mf = pyscf.scf.newton(mf).set(conv_tol=1e-9)
    mf.init_guess = guess
    mf
end

# QChem

function qchem_init_guess_Ti2O4(guess::String)
    pyqchem = pyimport("pyqchem")
    parser = pyimport("pyqchem.parsers.basic")
    mol = pyqchem.Structure(coordinates=[[1.730, 0.000, 0.000],
                                        [-1.730, 0.000, 0.000],
                                        [0.000, 1.224, 0.000],
                                        [0.000, -1.224, 0.000],
                                        [3.850, 0.000, 0.000],
                                        [-3.850, 0.000, 0.000]],
                            symbols=["Ti","Ti","O","O","O","O"],
                            charge=0,
                            multiplicity=3)
    qc_input = pyqchem.QchemInput(mol; max_scf_cycles=0, # Extract initial guess
                                  basis="def2-TZVP",     # chosen basis set
                                  scf_guess=guess,       # chosen guess
                                  # A priori the rest is useless but we never know...
                                  exchange="HF",
                                  # Correlation=None (default)
                                  unrestricted=false,
                                  scf_algorithm="DIIS",
                                  scf_convergence=8)
    output, data = pyqchem.get_output_from_qchem(qc_input; processors=4,
                                                 return_electronic_structure=true,
                                                 # parser=parser.basic_parser_qchem,
                                                 delete_scratch=false)

    # Find unitary rotation to go from Q-Chem to PySCF conventions
    # U*S_qchem*U' = S_pyscf
    n = size(data["overlap"],1)
    U = zeros(n,n)
    qchem_order = data["coefficients"]["qchem_order"]
    for i in 1:n
        id_i = only(findall(x->x==(i-1), order))
        U[i,id_i] = 1
    end
                    
   output, data, U
end
