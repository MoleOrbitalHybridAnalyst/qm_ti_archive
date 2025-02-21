from pyscf import gto, scf
import numpy as np
from sys import argv
import types

l = float(argv[1])
verbose = 0
sigma = 0.02
v_orb = 50

k_res = 0.01
a_res = 0

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

# g scales charges and v_res
def g0(l):
    return 1 - l
def dg0(l):
    return -1

def efv_scan(coords, box, init_dict=None):
    mol = gto.Mole()
    mol.atom = \
        '''
    C          -0.89307        0.58387       -0.11103
    C           0.64680        0.58527       -0.13528
    C          -1.41781       -0.71887        0.52089
    C          -1.41781        1.80558        0.66605
    H          -1.06978        1.77174        1.69389
    H          -2.50260        1.82306        0.67521
    H          -1.06706        2.72984        0.21897
    H          -1.06705       -1.58578       -0.02918
    H          -2.50260       -0.73728        0.52799
    H          -1.06977       -0.80306        1.54583
    H           1.02691       -0.26311       -0.69468
    H           1.03602        0.52709        0.87658
    H           1.02691        1.49217       -0.59375
    H          -1.25497        0.64263       -1.13293 '''
    mol.basis = '3-21g'
    mol._nelectron = 0
    mol.build()
    mol._nelectron = 33 + g0(l)
    mol.spin = 1 - g0(l)
    mol.tot_electrons = lambda *args: mol._nelectron
    mol.verbose = verbose
    mol.nelec = (17, 17 - mol.spin)
    mol.set_geom_(coords, unit='Bohr')
    
    # like we do in qmmm.mm_mole; works with pyscf_jojo qmmm branch and libcint v5.4.0
    nuc_charges = np.array(mol.atom_charges(), dtype=float)
    offset = mol._env.size
    mol._atm[:, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
    charges = nuc_charges.copy()
    charges[-1] *= g0(l)
    mol._env = np.append(mol._env, charges)
    mol._atm[:, gto.PTR_FRAC_CHARGE] = offset + np.arange(mol.natm)
    
    def fermi_occ(mu, mo_energy, sigma):
        occ = np.zeros_like(mo_energy)
        de = (mo_energy - mu) / sigma
        occ[de<40] = 1 / (np.exp(de[de<40])+1)
        return occ
    def dfermi_occ_dmu(mu, mo_energy, sigma):
        occ = np.zeros_like(mo_energy)
        de = (mo_energy - mu) / sigma
        occ[de<40] = np.exp(de[de<40]) / (np.exp(de[de<40])+1)**2 / sigma
        return occ
    def dfermi_occ_dsigma(mu, mo_energy, sigma):
        occ = np.zeros_like(mo_energy)
        de = (mo_energy - mu) / sigma
        occ[de<40] = np.exp(de[de<40]) / (np.exp(de[de<40])+1)**2 / sigma * de[de<40]
        return occ
    def get_occ(mf, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        na, nb = mol.nelec
        moe_a, moe_b = mo_energy
        def nelec_a(mu):
            return sum(fermi_occ(mu, moe_a, mf.sigma))
        def nelec_b(mu):
            return sum(fermi_occ(mu, moe_b, mf.sigma))
        from scipy.optimize import root
        sol = root(lambda mu: nelec_a(mu) - na, moe_a[max(0,int(na)-1)], tol=1e-12)
        mu_a = sol['x'][0]
        sol = root(lambda mu: nelec_b(mu) - nb, moe_b[max(0,int(nb)-1)], tol=1e-12)
        mu_b = sol['x'][0]
        mf.mu = mu_a, mu_b
        return fermi_occ(mu_a, moe_a, mf.sigma), fermi_occ(mu_b, moe_b, mf.sigma)
    
    mf = scf.UHF(mol)
    mf.sigma = sigma
    mf.init_guess = '1e'
    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)
    else:
        dm0 = None
    
    # add mu to ghost orbs
    ao_slices = mol.aoslice_by_atom()
    p0, p1 = ao_slices[-1][2:]
    h1mu = np.zeros([mol.nao]*2)
    np.fill_diagonal(h1mu[p0:p1,p0:p1], v_orb * f1(l))
    hcore = mf.get_hcore()
    mf.get_hcore = lambda self: hcore + h1mu
    
    mf.get_occ = types.MethodType(get_occ, mf)
    mf.max_cycle = 200
    E = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise Exception("SCF not converged")
    f = mf.mo_occ[0][(mf.mo_occ[0] >0) & (mf.mo_occ[0] < 1)]
    Selec = -(f @ np.log(f) + (1-f) @ np.log(1-f))
    f = mf.mo_occ[1][(mf.mo_occ[1] >0) & (mf.mo_occ[1] < 1)]
    Selec -= (f @ np.log(f) + (1-f) @ np.log(1-f))
    mf.entropy = Selec
    mf.free = mf.e_tot - mf.sigma * Selec
    E -= mf.sigma * Selec
    forces = -mf.nuc_grad_method().kernel()

    # restraints
    rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
    for i in range(mol.natm):
        rr[i, i] += 1e100
    def bond_energy_grad(r, k, a):
        rnorm = np.linalg.norm(r)
        E = k * (rnorm - a)**2
        g = 2 * k * (rnorm - a) / rnorm * r
        return E, g
    x12, y12, z12 = rr[1, 2]
    x02, y02, z02 = rr[0, 2]
    x01, y01, z01 = rr[0, 1]
    h_pos = mol.atom_coord(0)
    h_pos[0] += 1/4 * (y01 * z02 - z01 * y02)
    h_pos[1] += 1/4 * (z01 * x02 - x01 * z02)
    h_pos[2] += 1/4 * (x01 * y02 - y01 * x02)
    drX_dr0 = np.array([
            [4, z12,  -y12],
            [-z12, 4,  x12],
            [y12, -x12,  4]]) / 4
    drX_dr1 = np.array([
            [0.00, -z02,  y02],
            [z02,  0.00, -x02],
            [-y02,  x02,  0.00]]) / 4
    drX_dr2 = np.array([
            [0.00, z01,  -y01],
            [-z01, 0.00,  x01],
            [y01, -x01,  0.00]]) / 4
    Eres, gres = bond_energy_grad(mol.atom_coord(13)-h_pos, k_res, a_res)
    E += Eres * (1 - g0(l))
    forces[0]  += gres @ drX_dr0 * (1 - g0(l))
    forces[1]  += gres @ drX_dr1 * (1 - g0(l))
    forces[2]  += gres @ drX_dr2 * (1 - g0(l))
    forces[13] -= gres * (1 - g0(l))

    # dE / dl
    dEdl_nuc  = dg0(l) * mol.atom_charges() @ (nuc_charges[-1] / np.linalg.norm(rr[:, -1], axis=-1))

    dm = mf.make_rdm1()
    dhdl = 0
    with mol.with_rinv_origin(mol.atom_coord(-1)):
        dhdl -= mol.intor('int1e_rinv') * nuc_charges[-1] * dg0(l)
    dhmu = np.zeros_like(h1mu)
    np.fill_diagonal(dhmu[p0:p1,p0:p1], v_orb * df1(l))
    dEdl_elec = np.einsum('ij,xij->', (dhdl + dhmu), dm)
    dNe_dl = [0, dg0(l)]
    dEdl_elec += np.array(mf.mu) @ dNe_dl
    dEdl_elec -= mf.entropy * 0 # dsigma_dl(l)

    dEdl_res = -Eres * dg0(l)

    dEdl = dEdl_nuc + dEdl_elec + dEdl_res

    print("dEdl =", dEdl)
    init_dict = {'dm0': dm}

    return E, forces, None, init_dict

