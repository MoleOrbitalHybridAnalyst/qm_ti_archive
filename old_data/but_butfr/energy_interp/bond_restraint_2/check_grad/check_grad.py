from pyscf import gto, scf
import numpy as np
from sys import argv
import types

k_res = 0.01   # hartree / Bohr^2
a_res = 0.0   # Bohr
verbose = 0

# g scales Hamiltonian and v_res
def g0(l):
    return 1 - l
def dg0(l):
    return -1

def efv_scan(coords, box, l, init_dict=None):
    mol1 = gto.Mole()
    mol1.atom = \
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
    H          -1.25497        0.64263       -1.13293
    '''
    mol1.basis = '3-21g'
    mol1.build()
    mol1.set_geom_(coords, unit='Bohr')

    mol2 = gto.Mole()
    mol2.atom = \
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
    '''
    mol2.basis = '3-21g'
    mol2.spin = 1
    mol2.build()
    mol2.set_geom_(coords[:-1], unit='Bohr')

    mol1.verbose = verbose
    mol2.verbose = verbose
    
    mf1 = scf.UHF(mol1)
    mf2 = scf.UHF(mol2)
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
    x12, y12, z12 = rr[1, 2]
    x02, y02, z02 = rr[0, 2]
    x01, y01, z01 = rr[0, 1]
    h_pos = mol1.atom_coord(0)
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
    Eres, gres = bond_energy_grad(mol1.atom_coord(13)-h_pos, k_res, a_res)
    E += Eres * (1 - g0(l))
    forces[0]  += gres @ drX_dr0 * (1 - g0(l))
    forces[1]  += gres @ drX_dr1 * (1 - g0(l))
    forces[2]  += gres @ drX_dr2 * (1 - g0(l))
    forces[13] -= gres * (1 - g0(l))

    # dE / dl
    dEdl = (E1 - E2) * dg0(l)

    dm1 = mf1.make_rdm1()
    dm2 = mf2.make_rdm1()

    dEdl_res = -Eres * dg0(l)

    dEdl += dEdl_res

    init_dict = {'dm1': dm1, 'dm2': dm2, 'dEdl': dEdl}

    return E, forces, None, init_dict

if __name__ == "__main__":
    coords0 = np.array(\
      [[-1.68765771,  1.10335439, -0.20981629],
       [ 1.22227486,  1.10600001, -0.25564215],
       [-2.6792726 , -1.35846742,  0.98433944],
       [-2.6792726 ,  3.4120517 ,  1.25865209],
       [-2.02159121,  3.34810336,  3.20098819],
       [-4.7292286 ,  3.44508411,  1.27596198],
       [-2.01645116,  5.15864996,  0.41379333],
       [-2.01643226, -2.99668989, -0.05514221],
       [-4.7292286 , -1.39325728,  0.9977565 ],
       [-2.02157232, -1.51756346,  2.92119534],
       [ 1.94057865, -0.49720584, -1.31275494],
       [ 1.95779406,  0.99605574,  1.65649613],
       [ 1.94057865,  2.81979263, -1.12202489],
       [-2.37154959,  1.2143947 , -2.14092742]])

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
