from pyscf import gto, scf
import numpy as np
from sys import argv
import types

k_res = 0.01   # hartree / Bohr^2
v_orb = 500   # hartree
a = 0.0       # Bohr
verbose = 0

# f scaled v_orb
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

# g scales charge, v_res
def g0(l):
    return 1 - l
def dg0(l):
    return -1
def g1(l):
    return l
def dg1(l):
    return 1

def efv_scan(coords, box, l, init_dict=None):
    # ch3oh (l=0) <-> ch3f (l=1)
    mol = gto.Mole()
    mol.atom = \
    '''
C          -1.42203        2.29540       -0.02117
H          -1.09131        1.75525       -0.90230
H          -1.06708        1.77522        0.85454
H          -2.50707        2.30263       -0.00361
O          -0.86670        3.62335        0.04489
H          -1.14214        4.14086       -0.72261
X-F        -0.86670        3.62335        0.04489
    '''
    mol.basis = '3-21g'
    mol._nelectron = 18
    mol.build()
    mol.tot_electrons = lambda *args: mol._nelectron
    mol.verbose = verbose
    coords = np.vstack([coords, coords[-2:-1]])
    mol.set_geom_(coords, unit='Bohr')
    
    # like we do in qmmm.mm_mole; works with pyscf_jojo qmmm branch and libcint v5.4.0
    nuc_charges = np.array(mol.atom_charges(), dtype=float)
    nuc_charges[-1] = 9
    offset = mol._env.size
    mol._atm[:, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
    charges = nuc_charges.copy()
    charges[-3] = charges[-3] * g0(l) + nuc_charges[-1] * (1-g0(l))
    charges[-2] *= g0(l)
    charges[-1]  = 0
    mol._env = np.append(mol._env, charges)
    mol._atm[:, gto.PTR_FRAC_CHARGE] = offset + np.arange(mol.natm)
    
    mf = scf.RHF(mol)
    mf.init_guess = '1e'
    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)
    else:
        dm0 = None

    # add mu to ghost orbs
    ao_slices = mol.aoslice_by_atom()
    pm0, pm1 = ao_slices[-3][2], ao_slices[-2][3]
    pe0, pe1 = ao_slices[-1][2:]
    h1mu = np.zeros([mol.nao]*2)
    np.fill_diagonal(h1mu[pm0:pm1,pm0:pm1], v_orb * f1(l))
    np.fill_diagonal(h1mu[pe0:pe1,pe0:pe1], v_orb * f0(l))
    hcore = mf.get_hcore()
    mf.get_hcore = types.MethodType(lambda *args: hcore + h1mu, mf)

    mf.max_cycle = 200
    E = mf.kernel(dm0=dm0)
    if not mf.converged:
        raise Exception("SCF not converged")
    def grad_nuc(mf, mol=None, atmlst=None):
        '''
        Derivatives of nuclear repulsion energy wrt nuclear coordinates
        '''
        if mol is None:
            mol = mf.mol
        gs = np.zeros((mol.natm,3))
        for j in range(mol.natm):
            q2 = mol.atom_charge(j)
            r2 = mol.atom_coord(j)
            for i in range(mol.natm):
                if i != j:
                    q1 = mol.atom_charge(i)
                    if q1 * q2 == 0.0:
                        continue
                    r1 = mol.atom_coord(i)
                    r = np.sqrt(np.dot(r1-r2,r1-r2))
                    gs[j] -= q1 * q2 * (r2-r1) / r**3
        if atmlst is not None:
            gs = gs[atmlst]
        return gs
    mf_grad = mf.nuc_grad_method()
    mf_grad.grad_nuc = types.MethodType(grad_nuc, mf_grad)

    forces = -mf_grad.kernel()
    forces[-3] += forces[-1]
    forces = forces[:-1]

    # restraints
    rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
    rr[-3,-1] += 1e100
    rr[-1,-3] += 1e100
    for i in range(mol.natm):
        rr[i, i] += 1e100
    def bond_energy_grad(r, k, a):
        rnorm = np.linalg.norm(r)
        E = k * (rnorm - a)**2
        g = 2 * k * (rnorm - a) / rnorm * r
        return E, g
    h_pos = mol.atom_coord(4)
    h_pos += rr[4, 0] / 1.4
    Eres, gres = bond_energy_grad(mol.atom_coord(5)-h_pos, k_res, a)
    E += (Eres * g1(l))
    forces[4] += gres * g1(l) * (1/1.4 + 1)
    forces[0] -= gres * g1(l) / 1.4
    forces[5] -= gres * g1(l)

    # dE / dl
    dEdl_nuc  = dg0(l) * mol.atom_charges() @ ((nuc_charges[-3]-nuc_charges[-1]) / np.linalg.norm(rr[:, -3], axis=-1))
    dEdl_nuc += dg0(l) * mol.atom_charges() @ (nuc_charges[-2] / np.linalg.norm(rr[:, -2], axis=-1))

    dm = mf.make_rdm1()
    dhdl = 0
    with mol.with_rinv_origin(mol.atom_coord(-3)):
        dhdl -= mol.intor('int1e_rinv') * (nuc_charges[-3]-nuc_charges[-1]) * dg0(l)
    with mol.with_rinv_origin(mol.atom_coord(-2)):
        dhdl -= mol.intor('int1e_rinv') * nuc_charges[-2] * dg0(l)
    dhmu = np.zeros_like(h1mu)
    np.fill_diagonal(dhmu[pm0:pm1,pm0:pm1], v_orb * df1(l))
    np.fill_diagonal(dhmu[pe0:pe1,pe0:pe1], v_orb * df0(l))
    dEdl_elec = np.trace((dhdl + dhmu) @ dm)

    dEdl_res  = Eres * dg1(l)

    dEdl = dEdl_nuc + dEdl_elec + dEdl_res

#    print("dEdl =", dEdl)
    init_dict = {'dm0': dm, 'dEdl': dEdl}

    return E, forces, None, init_dict

if __name__ == "__main__":
    coords0 = \
    np.array([[-2.68724724e+00,  4.33767735e+00, -4.00055021e-02],
       [-2.06227702e+00,  3.31694178e+00, -1.70509988e+00],
       [-2.01648895e+00,  3.35467961e+00,  1.61484656e+00],
       [-4.73767568e+00,  4.35134007e+00, -6.82191131e-03],
       [-1.63782563e+00,  6.84713915e+00,  8.48298057e-02],
       [-2.15833180e+00,  7.82509132e+00, -1.36553499e+00]])

    l = 0.5
    e, f, _, d = efv_scan(coords0, np.eye(3)*1000, l)
    dEdl = d['dEdl']
    e1, _, _, _ = efv_scan(coords0, np.eye(3)*1000, l+5e-5)
    e2, _, _, _ = efv_scan(coords0, np.eye(3)*1000, l-5e-5)
    print(f"lambda = {l} numerical dEdl = {(e1-e2)/1e-4} analytical dEdl = {dEdl}")

    for i in range(len(coords0)):
        print(f"iatom = {i} numerical F = ", end='')
        for k in range(3):
            coords = coords0.copy()
            coords[i][k] -= 1e-4
            e1, _, _, _ = efv_scan(coords, np.eye(3)*1000, l)
            coords = coords0.copy()
            coords[i][k] += 1e-4
            e2, _, _, _ = efv_scan(coords, np.eye(3)*1000, l)
            print((e1-e2)/2e-4, end=' ')
        print(f"analytical F = {f[i][0]} {f[i][1]} {f[i][2]}")
