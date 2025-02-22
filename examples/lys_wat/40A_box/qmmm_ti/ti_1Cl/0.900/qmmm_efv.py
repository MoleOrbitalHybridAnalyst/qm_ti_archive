from gpu4qalchemify.qmmm.pbc.hf import merged_scf_generator
from pyscf import gto, lib, df
from gpu4pyscf.dft import RKS
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff
import numpy as np
import cupy as cp
from constants import *

from aspc_pred import DMPredictor

from sys import argv
from os import system, path, chdir, getcwd, environ

home_dir = environ['HOME']
l = float(argv[1])

# Chemistry - A European Journal, (2009), 186-197, 15(1)
# Shannon, “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides and Chalcogenides.” (Cl, K, Mg, Na)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'O': 0.63, 'CL': 1.67, 'K': 1.52, 'S': 1.03, 'P': 1.11, 'MG': 0.86, 'NA': 1.16}

# QM/MM setup
auxbasis = {'C': './6311Gss-rifit.dat', 'H': './6311Gss-rifit.dat', 'O': './6311Gss-rifit.dat', 'N': './6311Gss-rifit.dat', 'P': './6311Gss-rifit.dat', 'S': './6311Gss-rifit.dat', 'Mg': './6311Gss-rifit.dat'}
pseudo_bond_param_dir = f"{home_dir}/projects/pseudo_bond/jcp2008/refined_params/wb97x3c/separate"
aspc_nvec = 4
eval_cutoff_stride = 5
scf_conv_tol = 1e-10
scf_conv_tol_grad = 1e-6
max_scf_cycles = 1000
screen_tol = 1e-14
grids_level = 3
rcut_hcore0 = 20 * A / Bohr
rcut_ewald = 32 * A / Bohr
max_memory = 80000
verbose = 3

fp = open("./geom.xyz")
fp.readline(); fp.readline()
coords = np.array([line.split()[1:] for line in fp], dtype=float)
fp.close()
coords = coords * A / Bohr
box = np.diag([47.4661416415, 47.4661416415, 47.4661416415]) * A / Bohr
pseudo_bond_param_dir = f"/home/chhli/projects/pseudo_bond/jcp2008/refined_params/wb97x3c/separate"

k_res = 0.1  # hartree / Bohr^2
v_orb = 50   # hartree
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
def geom_pred(coords, return_grad=False):
    x = [-0.77047501,  3.77812947, -1.00382723, -1.00382723]
    h_pos  = coords[10] * x[0]
    h_pos += coords[13] * x[1]
    h_pos += coords[14] * x[2]
    h_pos += coords[15] * x[3]
    if not return_grad:
        return h_pos
    g = np.zeros((1,3,16,3))
    g[:,:,10,:] = x[0] * np.eye(3)
    g[:,:,13,:] = x[1] * np.eye(3)
    g[:,:,14,:] = x[2] * np.eye(3)
    g[:,:,15,:] = x[3] * np.eye(3)
    return g

def read_indexes(inp):
    ss = np.genfromtxt(inp, dtype=str)
    assert ss[0] == "group"
    assert ss[2] == "id"
    return np.array(ss[3:], dtype=int) - 1 # cuz lmp uses serial

qm_indexes = read_indexes("./group_qm.inp")                     # all qm atoms, including CA
zeroq_indexes = np.loadtxt("./zeroq_index.dat", dtype=int)      # mm charges invisible to qm
ca_indexes = read_indexes("./group_ca.inp")                     # pseudo_bond atoms
proton_index = 24                                               # proton index to be alchemically changed
cl_indexes = read_indexes("./group_cl.inp")
ca_resnames = ["LYS"]

with open("./topo.xyz") as fp:
    # let's assume topo.xyz only contains physical atoms
    fp.readline(); fp.readline()
    elements = np.array([line.split()[0] for line in fp])

with open("./equil.data") as fp:
    tags = list()
    charges = list()
    jline = np.inf
    for iline, line in enumerate(fp):
        if line[:5] == "Atoms":
            jline = iline
            assert line.split()[-1] == "full"
        if iline >= jline + 2:
            if line == "\n":
                break
            lsplt = line.split()
            tags.append(lsplt[0])
            charges.append(lsplt[3])
    tags = np.array(tags, dtype=int)
    charges = np.array(charges, dtype=float)
    order = np.argsort(tags)
    charges = charges[order]

def d4(mol, pidx_in_mol, l):
    # assume one lambda at this moment
    p = pidx_in_mol
    from dftd4.interface import DampingParam, DispersionModel
    charges0 = np.asarray(\
        [mol.atom_charge(i)+mol.atom_nelec_core(i) for i in range(mol.natm)])
    # protonated state
    charges = np.array(charges0, dtype=int)
    charges[p] = 1
    with lib.with_omp_threads(1):
        model = DispersionModel(
                charges,
                mol.atom_coords(),
                mol.charge)
        param = DampingParam(s6=1.0, s8=0.0, s9=1.0, a1=0.2464, a2=4.737)
        res1 = model.get_dispersion(param, grad=True)
        g1 = res1['gradient']
    # deprotonated state
    npidx_in_mol = [i for i in range(mol.natm) if i != pidx_in_mol]
    charges = np.array(charges0, dtype=int)[npidx_in_mol]
    with lib.with_omp_threads(1):
        model = DispersionModel(
                charges,
                mol.atom_coords()[npidx_in_mol],
                mol.charge-1)
        param = DampingParam(s6=1.0, s8=0.0, s9=1.0, a1=0.2464, a2=4.737)
        res0 = model.get_dispersion(param, grad=True)
        g0 = np.zeros_like(g1)
        g0[npidx_in_mol] = res0['gradient']
    E = (1-l) * res0['energy'] + l * res1['energy']
    grad = (1-l) * g0 + l * g1
    dhdl = res1['energy'] - res0['energy']
    return E, grad, dhdl

def efv_scan(coords, box, init_dict):
    '''
    return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
    '''

    natom = len(elements)                               # number of physical atoms
    
    coords_A = coords * Bohr / A
    box_A = box * Bohr / A
    mm_indexes = [i for i in range(natom) if not i in qm_indexes]
    
    ########################################
    ########### PySCF QM PART ##############
    ########################################
    
    # make qm atoms whole
    ref = coords_A[qm_indexes[0]]
    diff = coords_A[qm_indexes] - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A[qm_indexes] = diff + ref
    
    # move qm atoms to the center of box
    ref = np.mean(coords_A[qm_indexes], axis=0)
    diff = coords_A - ref
    n = (diff + 0.5 * np.diag(box_A)) // np.diag(box_A)
    diff = diff - n * np.diag(box_A)
    coords_A = diff
    
    for i, idx in enumerate(ca_indexes):
        # change elements[CA] into special F types
        elements[idx] = f"F{i}"
    pos2str = lambda pos: " ".join([str(x) for x in pos])
    atom_str = [f"{a} {pos2str(pos)}\n" \
            for a,pos in zip(elements[qm_indexes],coords_A[qm_indexes])]
    molB = gto.Mole()
    molB.atom = atom_str
    molB.charge = 1
    molB.spin = 0
    molB.basis = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/basis_vDZP_NWCHEM.dat'}
    molB.ecp = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/ecp_vDZP_NWCHEM.dat'}
    for i, resname in enumerate(ca_resnames):
            # basis and ecp for CA pseudo_bond
            fname = f"{pseudo_bond_param_dir}/{resname.lower()}/sto-2g.dat"
            molB.basis[f"F{i}"] = fname
            molB.ecp[f"F{i}"] = fname
    molB.build()
    
    qm_indexesA = list(qm_indexes)
    qm_indexesA.remove(proton_index)
    atom_str = [f"{a} {pos2str(pos)}\n" \
            for a,pos in zip(elements[qm_indexesA],coords_A[qm_indexesA])]
    molA = gto.Mole()
    molA.atom = atom_str
    molA.charge = 0
    molA.spin = 0
    molA.basis = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/basis_vDZP_NWCHEM.dat'}
    molA.ecp = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/ecp_vDZP_NWCHEM.dat'}
    for i, resname in enumerate(ca_resnames):
            # basis and ecp for CA pseudo_bond
            fname = f"{pseudo_bond_param_dir}/{resname.lower()}/sto-2g.dat"
            molA.basis[f"F{i}"] = fname
            molA.ecp[f"F{i}"] = fname
    molA.build()
    
    mm_coords = coords_A[mm_indexes] * A / Bohr
    mm_charges = charges.copy()     # all charges in tag order read from LMP data
    mm_charges[zeroq_indexes] = 0.0 # turn off backbone charges
    mm_charges[ca_indexes] = 0.0    # turn off CA charges in case not in zeroq_indexes
    mm_chargesA = mm_charges.copy()
    mm_chargesA[cl_indexes[0]] = 0.0
    mm_chargesA = mm_chargesA[mm_indexes]
    mm_chargesB = mm_charges.copy()
    mm_chargesB[cl_indexes[0]] = -1.0
    mm_chargesB = mm_chargesB[mm_indexes]
    mm_radii = np.array([ele2radius[e.upper()] for e in elements[mm_indexes]]) * A / Bohr
    assert abs(np.sum(mm_chargesA) + np.sum(molA.atom_charges()) - molA.nelectron) < 1e-8
    assert abs(np.sum(mm_chargesB) + np.sum(molB.atom_charges()) - molB.nelectron) < 1e-8
    
    auxbas = df.make_auxbasis(molB)
    if auxbasis is not None:
        for ele, bas in auxbasis.items():
            auxbas[ele] = bas
    def rks(mol):
        mf = RKS(mol, xc='wb97xv').density_fit(auxbasis=auxbas)
        mf.init_guess = '1e'
        mf.grids.level = grids_level
        mf.conv_tol = scf_conv_tol
        mf.conv_tol_grad = scf_conv_tol_grad
        mf.max_cycle = max_scf_cycles
        mf.screen_tol = screen_tol
        mf.conv_check = False
        mf._numint.libxc.is_nlc = lambda *args: False  # turn off VV10
        return mf
    
    singleA = list(range(16))
    singleB = list(range(15)) + [16]
    dualA = list()
    dualB = [15]
    mfgen = merged_scf_generator(
            rks,
            molA, molB,
            singleA,
            singleB,
            dualA,
            dualB,
            mm_coords,
            box,
            mm_chargesA,
            mm_chargesB,
            mm_radii=mm_radii,
            rcut_ewald=rcut_ewald,
            rcut_hcore=rcut_hcore0,
            fsw_mm_charge=lambda l, return_grad=False: [1-l,-1][return_grad],
            fsw_nelectron=None,
            fsw_spin=None,
            fsw_ham_single=None,
            fsw_ham_dualA=None,
            fsw_ham_dualB=lambda l, return_grad=False: [1-l,-1][return_grad],
            vorb_molA=None,
            vorb_molB=lambda l, return_grad=False: [v_orb*f0(l), v_orb*df0(l)][return_grad],
            geom_res_fc_dualA=None,
            geom_res_fc_dualB=lambda l, return_grad=False: [k_res*(1-l),-k_res][return_grad],
            geom_dualA_pred=None,
            geom_dualB_pred=geom_pred,
            )
    
    mf = mfgen(float(argv[1]))

    # get initial guess
    mf_ = rks(mf.mol)
    s1e = cp.asarray(mf_.get_ovlp())
    if 'dm_predictor' in init_dict:
        ps0 = init_dict['dm_predictor'].predict()
        mo0 = init_dict['mo0']
        mo0 = cp.dot(ps0, mo0)
        nocc = mf_.mol.nelectron // 2
        csc = cp.dot(cp.dot(mo0[:,:nocc].T, s1e), mo0[:,:nocc])
        w, v = cp.linalg.eigh(csc)
        csc_invhalf = cp.dot(v, cp.diag(w**(-0.5)) @ v.T)
        mo0[:,:nocc] = cp.dot(mo0[:,:nocc], csc_invhalf)
        mo_occ = mf_.get_occ(np.arange(mf_.mol.nao), mo0)
        dm0 = mf_.make_rdm1(mo0, mo_occ)
    else:
        w, v = mf_._eigh(cp.asarray(mf_.get_hcore()), s1e)
        dm0 = mf_.make_rdm1(v, mf_.get_occ(np.arange(s1e.shape[1]), v))
    # get rcut_hcore
    istep = init_dict.get('istep', 0)
    if dm0 is not None and \
            istep % eval_cutoff_stride == 1:
        rcut_hcore, _ = \
            determine_hcore_cutoff(
                mf_.mol, mm_coords, box, mf.mm_mol.atom_charges(), 15*A/Bohr,
                dm0.get(), rcut_step=1.0*A/Bohr, precision=2e-5, unit='Bohr')
        init_dict['rcut_hcore'] = rcut_hcore
        print(f"istep = {istep} rcut_hcore = {rcut_hcore}")
    else:
        rcut_hcore = init_dict.get("rcut_hcore", rcut_hcore0)

    mf.mm_mol.rcut_hcore = rcut_hcore
    mf.kernel(dm0=dm0)
    dEdl = mf.energy_tot_lgrad()
    dEmmdl = mf.energy_mm_lgrad()

    Edisp, dEdispdR, dEdispdl = d4(molB, dualB[0], l)

    # mf.e_tot = QM and QM-MM energy
    # energy_mm = electrostatic energy between alchemical MM charges and regular MM charges
    # disp = D4 dispersion as part of wB97X-3c
    E = mf.e_tot + mf.energy_mm() + Edisp

    # accordingly, dEdl has three contributions
    dEdl += dEmmdl + dEdispdl

    # nuc grad of QM + QM-MM energy
    mf_grad = mf.nuc_grad_method()
    mf_grad.max_memory = max_memory
    mf_grad.auxbasis_response = True
    dm = mf.make_rdm1()
    f = np.zeros((len(elements), 3))
    # NOTE that mf_grad.kernel() follows the atom order in mf.mol instead of molB
    # mf.mol order: molA, changeB (none here), dualB (H)
    fqm = -mf_grad.kernel()
    f[qm_indexes[singleB]] = fqm[singleA]
    f[qm_indexes[dualB]] = fqm[mf.mol.dualB]
    f[mm_indexes] -= (mf_grad.grad_nuc_mm() + mf_grad.grad_hcore_mm(dm) + mf_grad.de_ewald_mm)

    # nuc grad of energy_mm
    f[mm_indexes] -= mf.grad_mm()

    # nuc grad of Edisp
    f[qm_indexes] -= dEdispdR

    print("dEdl =", dEdl)

    # info for next steps
    if 'dm_predictor' not in init_dict:
        init_dict['dm_predictor'] = DMPredictor(aspc_nvec)
    init_dict['dm_predictor'].append(cp.dot(dm, s1e))
    init_dict['mo0'] = mf.mo_coeff
    init_dict['istep'] = istep + 1

    return E.get(), f, None, init_dict

