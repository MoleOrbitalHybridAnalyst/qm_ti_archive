from pyscf import gto, scf
import numpy as np
from sys import argv
import types

l = float(argv[1])
k_res = 0.01  # hartree / Bohr^2
v_orb = 50    # hartree
a = 0.0       # Bohr
verbose = 0

# g scales Hamiltonian and v_res
def g0(l):
    return 1 - l
def dg0(l):
    return -1
def g1(l):
    return l
def dg1(l):
    return 1

def efv_scan(coords, box, init_dict=None):
    # ch3oh (l=0) <-> ch3f (l=1)
    mol1 = gto.Mole()
    mol1.atom = \
    '''
C          -1.42203        2.29540       -0.02117
H          -1.09131        1.75525       -0.90230
H          -1.06708        1.77522        0.85454
H          -2.50707        2.30263       -0.00361
O          -0.86670        3.62335        0.04489
H          -1.14214        4.14086       -0.72261
    '''
    mol1.basis = '3-21g'
    mol1.build()
    mol1.set_geom_(coords, unit='Bohr')

    mol2 = gto.Mole()
    mol2.atom = \
    '''
C          -1.42203        2.29540       -0.02117
H          -1.09131        1.75525       -0.90230
H          -1.06708        1.77522        0.85454
H          -2.50707        2.30263       -0.00361
F          -0.86670        3.62335        0.04489
    '''
    mol2.basis = '3-21g'
    mol2.build()
    mol2.set_geom_(coords[:-1], unit='Bohr')

    mol1.verbose = verbose
    mol2.verbose = verbose
    
    mf1 = scf.RHF(mol1)
    mf2 = scf.RHF(mol2)
    if init_dict is not None:
        dm1 = init_dict.get('dm1', None)
        dm2 = init_dict.get('dm2', None)
    else:
        dm1 = None
        dm2 = None

    mf1.max_cycle = 200
    mf2.max_cycle = 200
    E1 = mf1.kernel(dm0=dm1)
    E2 = mf2.kernel(dm0=dm2)
    E = E1 * g0(l) + E2 * (1-g0(l))
    if not mf1.converged:
        raise Exception("SCF 1 not converged")
    if not mf2.converged:
        raise Exception("SCF 2 not converged")
    F1 = -mf1.nuc_grad_method().kernel()
    F2 = -mf2.nuc_grad_method().kernel()

    forces = F1 * g0(l)
    forces[:-1] += F2 * (1-g0(l))

    # restraints
    rr = mol1.atom_coords()[:,None,:] - mol1.atom_coords()[None]
    for i in range(mol1.natm):
        rr[i, i] += 1e100
    def bond_energy_grad(r, k, a):
        rnorm = np.linalg.norm(r)
        E = k * (rnorm - a)**2
        g = 2 * k * (rnorm - a) / rnorm * r
        return E, g
    h_pos = mol1.atom_coord(4)
    h_pos += rr[4, 0] / 1.4
    Eres, gres = bond_energy_grad(mol1.atom_coord(-1)-h_pos, k_res, a)
    E += (Eres * g1(l))
    forces[4]  += gres * g1(l) * (1/1.4 + 1)
    forces[0]  -= gres * g1(l) / 1.4
    forces[-1] -= gres * g1(l)

    # dE / dl
    dEdl = (E1 - E2) * dg0(l)

    dm1 = mf1.make_rdm1()
    dm2 = mf2.make_rdm1()

    dEdl_res  = Eres * dg1(l)

    dEdl += dEdl_res

    print("dEdl =", dEdl)
    init_dict = {'dm1': dm1, 'dm2': dm2}

    return E, forces, None, init_dict

