from pyscf import gto, scf
import numpy as np
from sys import argv
import types

from qalchemify.scf.hf import merged_scf_generator

l = float(argv[1])
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

def efv_scan(coords, box, init_dict=None):
    # ch3oh (l=0) <-> ch3f (l=1)
    molA = gto.Mole()
    molA.atom = \
    ''' 
C           -1.42203        2.29540       -0.02117
H           -1.09131        1.75525       -0.90230
H           -1.06708        1.77522        0.85454
H           -2.50707        2.30263       -0.00361
O           -0.86670        3.62335        0.04489
H           -1.14214        4.14086       -0.72261
    ''' 
    molA.basis = '3-21g'
    molA.build()
    molA.set_geom_(coords, unit='Bohr')
    molA.verbose = verbose

    molB = gto.Mole()
    molB.atom = \
    ''' 
C           -1.42203        2.29540       -0.02117
H           -1.09131        1.75525       -0.90230
H           -1.06708        1.77522        0.85454
H           -2.50707        2.30263       -0.00361
F           -0.86670        3.62335        0.04489
    ''' 
    molB.build()

    def geom_pred(x, return_grad=False):
        hpos = (x[4]-x[0])/1.4 + x[4]
        grad = np.zeros((1,3,5,3))
        grad[0,:,0,:] = -1/1.4 * np.eye(3)
        grad[0,:,4,:] = (1/1.4 + 1) * np.eye(3)
        if return_grad:
            return grad
        else:
            return hpos

    mf = merged_scf_generator(scf.hf.RHF, molA, molB, [0,1,2,3,4], [0,1,2,3,4], [5], [],
            fsw_nelectron=None, fsw_spin=None,
            fsw_ham_single=lambda l, g=False: [1-l,-1][g], fsw_ham_dualA=lambda l,g=False: [1-l,-1][g], fsw_ham_dualB=None,
            vorb_molA=lambda l,g=False: [v_orb*f1(l),v_orb*df1(l)][g], vorb_molB=lambda l,g=False: [v_orb*f0(l),v_orb*df0(l)][g],
            geom_res_fc_dualA=lambda l,g=False: [k_res*l,k_res][g], geom_res_fc_dualB=None,
            geom_dualA_pred=geom_pred)(l)
    mf.init_guess = '1e'

    if init_dict is not None:
        dm0 = init_dict.get('dm0', None)

    E = mf.kernel()
    dEdl = mf.energy_tot_lgrad()
    dEdR = mf.nuc_grad_method().kernel()[:-1] # the last atom is ghost F

    print("dEdl =", dEdl)

    init_dict['dm0'] = mf.make_rdm1()

    return E, -dEdR, None, init_dict

