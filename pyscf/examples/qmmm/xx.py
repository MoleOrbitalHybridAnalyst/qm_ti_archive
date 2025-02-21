#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run HF with background charges.
'''

import numpy
from pyscf import gto, scf, qmmm

mol = gto.M(atom='''
O       1.520  6.740  0.670
H       1.010  7.200  1.330
H       0.940  6.040  0.370
H       2.449  6.942  0.361
            ''',
            basis='3-21g',
            verbose=4, charge=1)

#coords = [[2.449,  6.942,  0.361]]
#charges = [1.0]

mf = scf.UHF(mol)
#mf = qmmm.mm_charge(mf, coords, charges)
mf.run()
