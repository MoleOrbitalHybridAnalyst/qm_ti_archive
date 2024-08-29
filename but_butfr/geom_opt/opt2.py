from pyscf import gto, scf
import numpy as np
from pyscf.geomopt import geometric_solver


mol = gto.Mole()
mol.charge = 0
mol.spin = 1
mol.fromfile('butfr_init.xyz')
mol.basis = '3-21g'
mol.build()

mf = scf.UHF(mol)
mol_eq = geometric_solver.optimize(mf)
mol_eq.tofile('butfr.xyz')
