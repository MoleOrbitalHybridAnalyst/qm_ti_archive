from pyscf import gto, scf
import numpy as np
from pyscf.geomopt import geometric_solver


mol = gto.Mole()
mol.fromfile('but_init.xyz')
mol.basis = '3-21g'
mol.charge = 0
mol.build()

mf = scf.RHF(mol)
mol_eq = geometric_solver.optimize(mf)
mol_eq.tofile('but.xyz')
