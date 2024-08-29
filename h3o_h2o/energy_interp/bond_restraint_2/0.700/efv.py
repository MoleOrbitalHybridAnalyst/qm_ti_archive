from pyscf import gto, scf
import numpy as np
from sys import argv

l = float(argv[1])
k_res = 0.01 # hartree/bohr^2
a_res = 0    # bohr

def efv_scan(coords, box, init_dict=None):
    '''
    d E / d lambda
    '''
    mol1 = gto.Mole()
    mol1.atom = \
    '''
           O -3.37765e+00  9.76485e-01  2.93603e-01
           H -1.97549e+00  1.41857e+00  5.28212e-01
           H -3.71513e+00  1.69673e-01  7.74414e-01
           H -4.05053e+00  1.65513e+00  4.50586e-01
    '''
    mol1.basis = 'cc-pvdz'
    mol1.charge = 1
    mol1.build()
    mol1.set_geom_(coords, unit='Bohr')
    mol1.verbose = 0

    mol2 = gto.Mole()
    mol2.atom = \
    '''
           O -3.37765e+00  9.76485e-01  2.93603e-01
           H -1.97549e+00  1.41857e+00  5.28212e-01
           H -3.71513e+00  1.69673e-01  7.74414e-01
    '''
    mol2.basis = 'cc-pvdz'
    mol2.charge = 0
    mol2.build()
    mol2.set_geom_(coords[:3], unit='Bohr')
    mol2.verbose = 0

    mf1 = scf.RHF(mol1)
    mf2 = scf.RHF(mol2)
    if init_dict is not None:
        dm1 = init_dict.get('dm1', None)
        dm2 = init_dict.get('dm2', None)
    E1 = mf1.kernel(dm0=dm1)
    E2 = mf2.kernel(dm0=dm2)
    E = l * E1 + (1-l) * E2

    rr = mol1.atom_coords()[:,None,:] - mol1.atom_coords()[None]
    for i in range(mol1.natm):
        rr[i, i] += 1e100

    force      = l * (-mf1.nuc_grad_method().kernel())
    force[:3] += (1-l) * (-mf2.nuc_grad_method().kernel())

    dm1 = mf1.make_rdm1()
    dm2 = mf2.make_rdm1()
    init_dict = {'dm1': dm1, 'dm2': dm2}

    # restraint
    def bond_energy_grad(r, k, a):
        rnorm = np.linalg.norm(r)
        E = k * (rnorm - a)**2
        g = 2 * k * (rnorm - a) / rnorm * r
        return E, g
    h_pos = (rr[0,1] + rr[0,2]) / 1.2 + mol1.atom_coord(0)
    Eres, gres = bond_energy_grad(mol1.atom_coord(-1)-h_pos, k_res, a_res)
    E += Eres * (1-l)
    dhdl_res = Eres * (-1)
    force[0]  += gres * (1-l) * (2/1.2 + 1)
    force[1]  -= gres * (1-l) * (1/1.2)
    force[2]  -= gres * (1-l) * (1/1.2)
    force[-1] -= gres * (1-l)

    print("dEdl =", E1 - E2 + dhdl_res)

    return E, force, None, init_dict

