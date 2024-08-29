from pyscf import gto, scf, mp
import numpy as np
from sys import argv

l = float(argv[1])
v_orb = 500
k_res = 0.01 # hartree/bohr^2
a_res = 0    # bohr

# f scales v_orb
def f0(l):
    # turn on in state 0
    return np.exp(-20 * l)
def f1(l):
    # turn on in state 1
    return np.exp(-20 * (1-l))
def df0(l):
    return -20 * np.exp(-20 * l)
def df1(l):
    return 20 * np.exp(-20 * (1-l))

def efv_scan(coords, box, init_dict=None):
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
    mol._nelectron = mol.nelectron
    mol.set_geom_(coords, unit='Bohr')
    mol.verbose = 0
    # like we do in qmmm.mm_mole; works with pyscf_jojo qmmm branch and libcint v5.4.0
    offset = mol._env.size
    mol._atm[:, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
    charges = np.asarray([8,1,1,l])
    mol._env = np.append(mol._env, charges)
    mol._atm[:, gto.PTR_FRAC_CHARGE] = offset + np.arange(mol.natm)
    mf = scf.RHF(mol)
    mf.init_guess = '1e'
    ao_slices = mol.aoslice_by_atom()
    p0, p1 = ao_slices[-1][2:]
    h1mu = np.zeros([mol.nao]*2)
    np.fill_diagonal(h1mu[p0:p1,p0:p1], v_orb*f0(l))
    hcore = mf.get_hcore()
    mf.get_hcore = lambda *args: hcore + h1mu
    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)
    E = mf.kernel(dm0=dm0)

    mcc = mp.MP2(mf)
    Ecorr, t2 = mcc.kernel()
    E += Ecorr

    # relaxed density computed in mp2.grad_elec written to dm1.npy
    force = -mcc.nuc_grad_method().kernel()

    dm = np.load('dm1.npy')
    with mol.with_rinv_origin(mol.atom_coord(-1)):
        dhdl = -mol.intor('int1e_rinv')
    dhdmu = np.zeros([mol.nao]*2)
    np.fill_diagonal(dhdmu[p0:p1,p0:p1], v_orb * df0(l))
    dhdl_elec = np.trace((dhdl + dhdmu) @ dm)

    rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
    for i in range(mol.natm):
        rr[i, i] += 1e100
    dhdl_nuc = mol.atom_charges() @ (1 / np.linalg.norm(rr[:,-1], axis=-1))

    init_dict = {'dm0': mf.make_rdm1()}

    # restraint
    def bond_energy_grad(r, k, a):
        rnorm = np.linalg.norm(r)
        E = k * (rnorm - a)**2
        g = 2 * k * (rnorm - a) / rnorm * r
        return E, g
    h_pos = (rr[0,1] + rr[0,2]) / 1.2 + mol.atom_coord(0)
    Eres, gres = bond_energy_grad(mol.atom_coord(-1)-h_pos, k_res, a_res)
    E += Eres * (1-l)
    dhdl_res = Eres * (-1)
    force[0]  += gres * (1-l) * (2/1.2 + 1)
    force[1]  -= gres * (1-l) * (1/1.2)
    force[2]  -= gres * (1-l) * (1/1.2)
    force[-1] -= gres * (1-l)

    print("dEdl =", dhdl_nuc + dhdl_elec + dhdl_res)

    return E, force, None, init_dict

