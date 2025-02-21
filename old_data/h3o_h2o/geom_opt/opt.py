from pyscf import gto, scf
import numpy as np
from pyscf.geomopt import geometric_solver


mol = gto.Mole()
mol.atom = \
'''
       O -3.37765e+00  9.76485e-01  2.93603e-01
       H -1.97549e+00  1.41857e+00  5.28212e-01
       H -3.71513e+00  1.69673e-01  7.74414e-01
       H -4.05053e+00  1.65513e+00  4.50586e-01
'''
mol.basis = 'cc-pvdz'
mol.charge = 1
mol.build()

mf = scf.RHF(mol)
mol_eq = geometric_solver.optimize(mf)
mol_eq.tofile('h3o.xyz')
