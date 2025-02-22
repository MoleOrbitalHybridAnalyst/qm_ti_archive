from pyscf import gto, scf
import numpy as np
from sys import argv

from qalchemify.scf.hf import merged_scf_generator

l = float(argv[1])
v_orb = 50
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
    molA = gto.Mole()
    molA.atom = \
    '''
           O -3.37765e+00  9.76485e-01  2.93603e-01
           H -1.97549e+00  1.41857e+00  5.28212e-01
           H -3.71513e+00  1.69673e-01  7.74414e-01
    '''
    molA.basis = 'cc-pvdz'
    molA.charge = 0
    molA.build()
    molA.set_geom_(coords[:3], unit='Bohr')
    molA.verbose = 0

    molB = gto.Mole()
    molB.atom = \
    '''
           O -3.37765e+00  9.76485e-01  2.93603e-01
           H -1.97549e+00  1.41857e+00  5.28212e-01
           H -3.71513e+00  1.69673e-01  7.74414e-01
           H -4.05053e+00  1.65513e+00  4.50586e-01
    '''
    molB.charge = 1
    molB.build()
    molB.set_geom_(coords, unit='Bohr')
    molB.verbose = 0

    def geom_pred(x, return_grad=False):
        hpos = (2*x[0]-x[1]-x[2])/1.2+x[0]
        g = np.zeros((1,3,3,3))
        g[0,:,0,:] = np.eye(3) * (2/1.2+1)
        g[0,:,1,:] = np.eye(3) * (-1/1.2)
        g[0,:,2,:] = np.eye(3) * (-1/1.2)
        if return_grad:
            return g
        else:
            return hpos
    mf = merged_scf_generator(scf.hf.RHF, molA, molB, [0,1,2], [0,1,2], [], [3],
            fsw_nelectron=None, fsw_spin=None,
            fsw_ham_single=None, fsw_ham_dualA=None, fsw_ham_dualB=lambda l, return_grad=False: [1-l,-1][return_grad],
            vorb_molA=None, vorb_molB=lambda l, return_grad=False: [v_orb*f0(l), v_orb*df0(l)][return_grad],
            geom_res_fc_dualA=None, geom_res_fc_dualB=lambda l, return_grad=False: [k_res*(1-l),-k_res][return_grad],
            geom_dualA_pred=None, geom_dualB_pred=geom_pred)(l)

    mf.init_guess = '1e'

    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)

    E = mf.kernel(dm0=dm0)
    dEdl = mf.energy_tot_lgrad()
    dEdR = mf.nuc_grad_method().kernel()

    print("dEdl =", dEdl)

    init_dict['dm0'] = mf.make_rdm1()

    return E, -dEdR, None, init_dict

