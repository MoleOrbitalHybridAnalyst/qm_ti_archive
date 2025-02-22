from pyscf import gto, scf
import numpy as np
from sys import argv
import types

from qalchemify.scf.hf import merged_scf_generator

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
    # butane -> butyl
    molA = gto.Mole()
    molA.atom = \
         '''
    C           -0.89307        0.58387       -0.11103
    C            0.64680        0.58527       -0.13528
    C           -1.41781       -0.71887        0.52089
    C           -1.41781        1.80558        0.66605
    H           -1.06978        1.77174        1.69389
    H           -2.50260        1.82306        0.67521
    H           -1.06706        2.72984        0.21897
    H           -1.06705       -1.58578       -0.02918
    H           -2.50260       -0.73728        0.52799
    H           -1.06977       -0.80306        1.54583
    H            1.02691       -0.26311       -0.69468
    H            1.03602        0.52709        0.87658
    H            1.02691        1.49217       -0.59375
    H           -1.25497        0.64263       -1.13293 '''
    molA.basis = '3-21g'
    molA.build()
    molA.verbose = verbose
    molA.set_geom_(coords, unit='Bohr')

    molB = gto.Mole()
    molB.atom = \
         '''
    C           -0.89307        0.58387       -0.11103
    C            0.64680        0.58527       -0.13528
    C           -1.41781       -0.71887        0.52089
    C           -1.41781        1.80558        0.66605
    H           -1.06978        1.77174        1.69389
    H           -2.50260        1.82306        0.67521
    H           -1.06706        2.72984        0.21897
    H           -1.06705       -1.58578       -0.02918
    H           -2.50260       -0.73728        0.52799
    H           -1.06977       -0.80306        1.54583
    H            1.02691       -0.26311       -0.69468
    H            1.03602        0.52709        0.87658
    H            1.02691        1.49217       -0.59375 '''
    molB.spin = 1
    molB.build()
    molB.verbose = verbose
    
    cg0 = lambda l,g=False: [g0(l), dg0(l)][g]
    def geom_pred(x, return_grad=False):
        x12, y12, z12 = x[2] - x[1]
        x02, y02, z02 = x[2] - x[0]
        x01, y01, z01 = x[1] - x[0]
        h_pos = x[0]
        h_pos[0] += 1/4 * (y01 * z02 - z01 * y02)
        h_pos[1] += 1/4 * (z01 * x02 - x01 * z02)
        h_pos[2] += 1/4 * (x01 * y02 - y01 * x02)
        if not return_grad:
            return h_pos
        else:
            pass
        g = np.zeros((1,3,13,3))
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
        g[:,:,0,:] = drX_dr0.T
        g[:,:,1,:] = drX_dr1.T
        g[:,:,2,:] = drX_dr2.T
        return g
    mf = merged_scf_generator(scf.uhf.UHF, molA, molB, list(range(13)), list(range(13)), [13], [],
            fsw_nelectron=cg0, fsw_spin=cg0, sigma=lambda l,g=False: [sigma,0][g],
            fsw_ham_single=None, fsw_ham_dualA=cg0, fsw_ham_dualB=None,
            vorb_molA=lambda l,g=False: [v_orb*f1(l),v_orb*df1(l)][g], vorb_molB=None,
            geom_res_fc_dualA=lambda l,g=False: [k_res*(1-g0(l)),-k_res*dg0(l)][g], geom_res_fc_dualB=None,
            geom_dualA_pred=geom_pred, geom_dualB_pred=None)(l)
    mf.init_guess = '1e'

    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)

    E = mf.kernel()
    dEdl = mf.energy_tot_lgrad()
    dEdR = mf.nuc_grad_method().kernel()

    print("dEdl =", dEdl)

    init_dict['dm0'] = mf.make_rdm1()

    return E, -dEdR, None, init_dict

