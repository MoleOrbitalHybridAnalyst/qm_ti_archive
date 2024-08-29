from pyscf import gto
from pyscf import df, lib
from gpu4pyscf.qmmm.pbc import itrf, mm_mole
from gpu4pyscf.qmmm.pbc.tools import determine_hcore_cutoff
from gpu4pyscf.dft import RKS
from gpu4pyscf.df import int3c2e
from gpu4pyscf.lib import cupy_helper

from scipy.special import erf

from aspc_pred import DMPredictor

import numpy as np
import cupy as cp
from constants import *

from time import time
from sys import argv
from os import system, path, chdir, getcwd, environ

home_dir = environ['HOME']
lambdas = [float(argv[1])]


# Chemistry - A European Journal, (2009), 186-197, 15(1)
# Shannon, “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides and Chalcogenides.” (Cl, K, Mg, Na)
ele2radius = {'N': 0.71, 'H': 0.32, 'C': 0.75, 'O': 0.63, 'CL': 1.67, 'K': 1.52, 'S': 1.03, 'P': 1.11, 'MG': 0.86, 'NA': 1.16}

charge = 0
spin = 0

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
rcut_hcore0 = 20          # in angstrom
rcut_ewald = 32           # in angstrom
max_memory = 80000
verbose = 3

# alchemical setup
from my_restraint import compute_restraint
v_orb = 50
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

def read_indexes(inp):
    ss = np.genfromtxt(inp, dtype=str)
    assert ss[0] == "group"
    assert ss[2] == "id"
    return np.array(ss[3:], dtype=int) - 1 # cuz lmp uses serial

qm_indexes = read_indexes("./group_qm.inp")                     # all qm atoms, including CA
zeroq_indexes = np.loadtxt("./zeroq_index.dat", dtype=int)      # mm charges invisible to qm
ca_indexes = read_indexes("./group_ca.inp")                     # pseudo_bond atoms
#proton_indexes = np.loadtxt("./proton_indexes.dat", dtype=int)  # proton indexes to be alchemically changed
proton_indexes = [16]
cl_indexes = read_indexes("./group_cl.inp")
ca_resnames = ["ASP"]

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

def d4(mol, pidx_in_mol, lambdas):
    # assume one lambda at this moment
    assert len(pidx_in_mol) == 1
    assert len(lambdas) == 1
    p, l = pidx_in_mol[0], lambdas[0]
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
    npidx_in_mol = [i for i in range(mol.natm) if i not in pidx_in_mol]
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
    dhdl = [res1['energy'] - res0['energy']]
    return E, grad, dhdl

def select_mm_atoms(mf):
    mol = mf.mol
    mm_mol = mf.mm_mol
    Ls = mm_mol.get_lattice_Ls()
    qm_center = np.mean(mol.atom_coords(), axis=0)
    all_coords = lib.direct_sum('ix+Lx->Lix', 
            mm_mol.atom_coords(), Ls).reshape(-1,3)
    all_charges = np.hstack([mm_mol.atom_charges()] * len(Ls))
    all_expnts = np.hstack([np.sqrt(mm_mol.get_zetas())] * len(Ls))
    dist2 = all_coords - qm_center
    dist2 = lib.einsum('ix,ix->i', dist2, dist2)
    mask = dist2 <= mm_mol.rcut_hcore**2
    charges = all_charges[mask]
    coords = all_coords[mask]
    expnts = all_expnts[mask]
    return charges, coords, expnts

def get_ewald_hess(mm_mol, coords1, coords2):
    '''
    modified gpu4pyscf.qmmm.pbc.mm_mole.get_ewald_pot
    such that real-space Coulomb within rcut_hcore is not subtracted
    and diff. in Gaussian and pc is not computed
    and Tijab not computed
    and charges2 are not contracted
    '''
    from gpu4pyscf.lib import cupy_helper
    from cupyx.scipy.special import erfc
    self = mm_mol
    assert self.dimension == 3
    coords1 = cp.asarray(coords1)
    coords2 = cp.asarray(coords2)

    ew_eta, ew_cut = self.get_ewald_params()
    mesh = self.mesh

    # TODO Lall should respect ew_rcut
    Lall = cp.asarray(self.get_lattice_Ls())

    all_coords2 = (coords2[None,:,:] - Lall[:,None,:]).reshape(-1,3)
    all_coords2 = cp.asarray(all_coords2)
    dist2 = all_coords2 - cp.mean(coords1, axis=0)[None]
    dist2 = cp.einsum('jx,jx->j', dist2, dist2)

    ewovrl00 = cp.zeros((len(coords1), len(coords2))) 
    ewovrl01 = cp.zeros((len(coords1), len(coords2), 3)) 
    ewself00 = cp.zeros((len(coords1), len(coords2))) 
    ewself01 = cp.zeros((len(coords1), len(coords2), 3)) 

    mem_avail = cupy_helper.get_avail_mem()
    blksize = int(mem_avail/64/3/len(all_coords2))
    if blksize == 0:
        raise RuntimeError(f"Not enough GPU memory, mem_avail = {mem_avail}, blkszie = {blksize}")
    for i0, i1 in lib.prange(0, len(coords1), blksize):
        R = coords1[i0:i1,None,:] - all_coords2[None,:,:]
        r = cp.linalg.norm(R, axis=-1)
        r[r<1e-16] = 1e100
        rmax_qm = max(cp.linalg.norm(coords1 - cp.mean(coords1, axis=0), axis=-1))

        # ewald real-space sum
        # ewald sum will run over all Lall images regardless of ew_cut
        # this is to ensure r and R will always have the shape of (i1-i0, L*num_qm)
        r_ = r
        R_ = R
        ekR = cp.exp(-ew_eta**2 * r_**2)
        Tij = erfc(ew_eta * r_) / r_
        invr3 = (Tij + 2*ew_eta/cp.sqrt(cp.pi) * ekR) / r_**2
        Tija = -cp.einsum('ijx,ij->ijx', R_, invr3)

        Tij = cp.sum(Tij.reshape(i1-i0, len(Lall), len(coords2)), axis=1)
        Tija = cp.sum(Tija.reshape(i1-i0, len(Lall), len(coords2), 3), axis=1)
        ewovrl00[i0:i1] += Tij
        ewovrl01[i0:i1] += Tija
        ekR = Tij = invr3 = None

        r_ = R_ = None

    R = r = dist2 = None

    # g-space sum (using g grid)

    Gv, Gvbase, weights = self.get_Gv_weights(mesh)
    absG2 = cp.einsum('gx,gx->g', Gv, Gv)
    absG2[absG2==0] = 1e200

    coulG = 4*cp.pi / absG2
    coulG *= weights
    # NOTE Gpref is actually Gpref*2
    Gpref = cp.exp(-absG2/(4*ew_eta**2)) * coulG

    GvR2 = cp.einsum('gx,ix->ig', Gv, coords2)
    cosGvR2 = cp.cos(GvR2)
    sinGvR2 = cp.sin(GvR2)
    GvR1 = cp.einsum('gx,ix->ig', Gv, coords1)
    cosGvR1 = cp.cos(GvR1)
    sinGvR1 = cp.sin(GvR1)

    # qm pc - qm pc
    ewg00  = cp.einsum('ig,jg,g->ij', cosGvR1, cosGvR2, Gpref)
    ewg00 += cp.einsum('ig,jg,g->ij', sinGvR1, sinGvR2, Gpref)
    # qm pc - qm dip
    ewg01  = -cp.einsum('gx,ig,jg,g->ijx', Gv, sinGvR1, cosGvR2, Gpref)
    ewg01 +=  cp.einsum('gx,ig,jg,g->ijx', Gv, cosGvR1, sinGvR2, Gpref)

    return ewovrl00 + ewself00 + ewg00, \
            ewovrl01 + ewself01 + ewg01

def efv_scan(coords, box, init_dict):
    '''
    return energy, force, virial given atom coords[N][3] and box[3][3] in Bohr
    '''
    t0 = time()

    # let's figure out some basics
    natom = len(elements)                               # number of physical atoms
    nlambdas = len(proton_indexes)                      # number of lambdas
    pidx_in_mol = \
        [list(qm_indexes).index(p) for p in proton_indexes]

    coords_A = coords * Bohr / A
    box_A = box * Bohr / A
    mm_indexes = [i for i in range(natom) if not i in qm_indexes]
    clidx_in_mm = \
        [mm_indexes.index(cl) for cl in cl_indexes]

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

    t1 = time()

    for i, idx in enumerate(ca_indexes):
        # change elements[CA] into special F types
        elements[idx] = f"F{i}"
    pos2str = lambda pos: " ".join([str(x) for x in pos])
    atom_str = [f"{a} {pos2str(pos)}\n" \
            for a,pos in zip(elements[qm_indexes],coords_A[qm_indexes])]
    mol = gto.Mole()
    mol.atom = atom_str
    mol.charge = charge
    mol.spin = spin
    mol.verbose = verbose
    mol.max_memory = max_memory 
    mol.basis = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/basis_vDZP_NWCHEM.dat'}
    mol.ecp = {'default': f'{home_dir}/projects/wb97x-3c/jcp2023/supporting_info/vDZP_basis/ecp_vDZP_NWCHEM.dat'}
    for i, resname in enumerate(ca_resnames):
        # basis and ecp for CA pseudo_bond
        fname = f"{pseudo_bond_param_dir}/{resname.lower()}/sto-2g.dat"
        mol.basis[f"F{i}"] = fname
        mol.ecp[f"F{i}"] = fname
    mol.build()
    print("mol.nao =", mol.nao)

    # now alchemically modify mol
    mol._nelectron = mol.nelectron
    org_nuc_charges = mol.atom_charges().copy()
    nuc_charges = np.array(mol.atom_charges(), dtype=float)
    offset = mol._env.size
    mol._atm[:, gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
    for p, l in zip(pidx_in_mol, lambdas):
        nuc_charges[p] *= l
    mol._env = np.append(mol._env, nuc_charges)
    mol._atm[:, gto.PTR_FRAC_CHARGE] = offset + np.arange(mol.natm)

    mm_coords = coords_A[mm_indexes]
    mm_charges = charges.copy()     # all charges in tag order read from LMP data
    mm_charges[zeroq_indexes] = 0.0 # turn off backbone charges
    mm_charges[ca_indexes] = 0.0    # turn off CA charges in case not in zeroq_indexes
    for l, cl in zip(lambdas, cl_indexes):
        mm_charges[cl] *= l
    mm_charges = mm_charges[mm_indexes]
    mm_radii = [ele2radius[e.upper()] for e in elements[mm_indexes]]
    assert abs(np.sum(mm_charges) + np.sum(mol.atom_charges()) - mol.nelectron) < 1e-8

    auxbas = df.make_auxbasis(mol)
    if auxbasis is not None:
        for ele, bas in auxbasis.items():
            auxbas[ele] = bas
    mf = RKS(mol, xc='wb97xv').density_fit(auxbasis=auxbas)
    mf.grids.level = grids_level
    mf.conv_tol = scf_conv_tol
    mf.conv_tol_grad = scf_conv_tol_grad
    mf.max_cycle = max_scf_cycles
    mf.screen_tol = screen_tol
    mf.conv_check = False
    mf._numint.libxc.is_nlc = lambda *args: False  # turn off VV10

    s1e = cp.asarray(mf.get_ovlp())
    if 'dm_predictor' in init_dict:
        ps0 = init_dict['dm_predictor'].predict()
        mo0 = init_dict['mo0']
        mo0 = cp.dot(ps0, mo0)
        nocc = mol.nelectron // 2
        csc = cp.dot(cp.dot(mo0[:,:nocc].T, s1e), mo0[:,:nocc])
        w, v = cp.linalg.eigh(csc)
        csc_invhalf = cp.dot(v, cp.diag(w**(-0.5)) @ v.T)
        mo0[:,:nocc] = cp.dot(mo0[:,:nocc], csc_invhalf)
        mo_occ = mf.get_occ(np.arange(mol.nao), mo0)
        dm0 = mf.make_rdm1(mo0, mo_occ)
    else:
        w, v = mf._eigh(cp.asarray(mf.get_hcore()), cp.asarray(mf.get_ovlp()))
        dm0 = mf.make_rdm1(v, mf.get_occ(np.arange(mol.nao), v))

    istep = init_dict.get('istep', 0)
    if dm0 is not None and \
            istep % eval_cutoff_stride == 1:
        rcut_hcore, _ = \
            determine_hcore_cutoff(
                mol, mm_coords, box_A, mm_charges, 15,
                dm0.get(), rcut_step=1.0, precision=2e-5)
        init_dict['rcut_hcore'] = rcut_hcore
        print(f"istep = {istep} rcut_hcore = {rcut_hcore}")
    else:
        rcut_hcore = init_dict.get("rcut_hcore", rcut_hcore0)

    # now mf is aware of MM charges
    mf = itrf.add_mm_charges(
        mf, mm_coords, box_A, mm_charges, mm_radii, 
        rcut_hcore=rcut_hcore, rcut_ewald=rcut_ewald)

    # changes for alchemistry
    ao_slices = mol.aoslice_by_atom()
    h1mu = np.zeros([mol.nao]*2)
    for p, l in zip(pidx_in_mol, lambdas):
        p0, p1 = ao_slices[p][2:]
        np.fill_diagonal(h1mu[p0:p1,p0:p1], v_orb*f0(l))
    hcore = mf.get_hcore()
    mf.get_hcore = lambda self: hcore + cp.asarray(h1mu)

    e_qmmm = mf.kernel(dm0=dm0)

    t2 = time()
    print("PySCF energy time =", t2 - t1)

    mf_grad = mf.nuc_grad_method()
    mf_grad.max_memory = max_memory
    mf_grad.auxbasis_response = True
    f_qmmm = np.zeros_like(coords)
    f_qmmm[qm_indexes] = -mf_grad.kernel()
    dm = mf.make_rdm1()
    f_qmmm[mm_indexes] = -(mf_grad.grad_nuc_mm() + mf_grad.grad_hcore_mm(dm) + mf_grad.de_ewald_mm)

    if 'dm_predictor' not in init_dict:
        init_dict['dm_predictor'] = DMPredictor(aspc_nvec)

    init_dict['dm_predictor'].append(cp.dot(dm, s1e))
    init_dict['mo0'] = mf.mo_coeff
    init_dict['istep'] = istep + 1

    t3 = time()
    print("PySCF grad time =", t3 - t2)

    # lambda gradient 
    mm_mol = mf.mm_mol
    qm_center = np.mean(mol.atom_coords(), axis=0)
    Ls = mm_mol.get_lattice_Ls()
    ## elec
    dhdl_elec = list()
    for p in pidx_in_mol:
        with mol.with_rinv_origin(mol.atom_coord(p)):
            dhdl = -org_nuc_charges[p] * mol.intor('int1e_rinv')
        dhdmu = np.zeros([mol.nao]*2)
        p0, p1 = ao_slices[p][2:]
        np.fill_diagonal(dhdmu[p0:p1,p0:p1], v_orb * df0(l))
        dhdl_elec.append(np.trace((dhdl + dhdmu) @ dm.get()))
    for icl, cl in enumerate(clidx_in_mm):
        q2, r2, expt = -1.0, mm_mol.atom_coord(cl), mm_mol.get_zetas()[cl]
        for L in Ls:
            dist = lib.norm(r2 + L - qm_center)
            if dist > mm_mol.rcut_hcore:
                continue
            else:
                fakemol = gto.fakemol_for_charges(
                        np.asarray([r2 + L]), np.asarray([expt]))

                intopt = int3c2e.VHFOpt(mol, fakemol, 'int2e')
                intopt.build(mf.direct_scf_tol, diag_block_with_triu=True, aosym=True, 
                             group_size=int3c2e.BLKSIZE, group_size_aux=int3c2e.BLKSIZE)

                nao_sph = len(intopt.sph_ao_idx)
                v = 0
                for cp_kl_id, _ in enumerate(intopt.aux_log_qs):
                    k0 = intopt.sph_aux_loc[cp_kl_id]
                    k1 = intopt.sph_aux_loc[cp_kl_id+1]
                    j3c = cp.zeros([k1-k0, nao_sph, nao_sph], order='C')
                    for cp_ij_id, _ in enumerate(intopt.log_qs):
                        cpi = intopt.cp_idx[cp_ij_id]
                        cpj = intopt.cp_jdx[cp_ij_id]
                        li = intopt.angular[cpi]
                        lj = intopt.angular[cpj]
                        int3c_blk = int3c2e.get_int3c2e_slice(intopt, cp_ij_id, cp_kl_id, omega=0.0)
                        int3c_blk = int3c2e.cart2sph(int3c_blk, axis=1, ang=lj)
                        int3c_blk = int3c2e.cart2sph(int3c_blk, axis=2, ang=li)
                        i0, i1 = intopt.sph_ao_loc[cpi], intopt.sph_ao_loc[cpi+1]
                        j0, j1 = intopt.sph_ao_loc[cpj], intopt.sph_ao_loc[cpj+1]
                        j3c[:,j0:j1,i0:i1] = int3c_blk
                        if cpi != cpj and intopt.aosym:
                            j3c[:,i0:i1,j0:j1] = int3c_blk.transpose([0,2,1])
                    v += cp.einsum('kji,k->ji', j3c, cp.asarray([-q2]))
                dhdl = cupy_helper.take_last2d(v, intopt.rev_ao_idx)
                intopt = int3c_blk = None
                dhdl_elec[icl] += cp.einsum('ij,ji->', dhdl, dm).get()
    ## nuc
    ### gas-phase nuc
    rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
    for i in range(mol.natm):
        rr[i, i] += 1e100
    dhdl_nuc = list()
    for p in pidx_in_mol:
        dhdl_nuc.append(
            mol.atom_charges() @ \
            (org_nuc_charges[p] / np.linalg.norm(rr[:,p], axis=-1)))
    ### qm_nuc - mm_pc
    chgs, crds, expts = select_mm_atoms(mf)
    for ip, p in enumerate(pidx_in_mol):
        q2, r2 = org_nuc_charges[p], mol.atom_coord(p)
        r = lib.norm(r2-crds, axis=1)
        dhdl_nuc[ip] += q2*(chgs * erf(expts*r) /r).sum()
    crds = mol.atom_coords()
    chgs = mol.atom_charges()
    for i, cl in enumerate(clidx_in_mm):
        q2, r2, expt = -1.0, mm_mol.atom_coord(cl), np.sqrt(mm_mol.get_zetas()[cl])
        for L in Ls:
            dist = lib.norm(r2 + L - qm_center)
            if dist > mm_mol.rcut_hcore:
                continue
            else:
                r = lib.norm(r2 + L - crds, axis=1)
                dhdl_nuc[i] += q2*(chgs * erf(expt*r) /r).sum()
    ## restraint
    Eres, gres, dhdl_res = compute_restraint(mol, pidx_in_mol, lambdas)
    e_qmmm += Eres
    f_qmmm[qm_indexes] -= gres
    ## Ewald
    qm_ewald_pot = mf.get_qm_ewald_pot(mol, dm, mf.qm_ewald_hess)
    ewald_pot = mf.mm_ewald_pot[0] + qm_ewald_pot[0]
    dhdl_ewald = list()
    for p in pidx_in_mol:
        dhdl_ewald.append(org_nuc_charges[p] * ewald_pot[p].get())
    for i, cl in enumerate(clidx_in_mm):
        # qm-mm ewald pot induced by d q_cl_i / d lambda
        mm_pot = mm_mol.get_ewald_pot(mol.atom_coords(), 
                    mm_mol.atom_coords()[[cl]],
                    cp.array([-1.0]),
                    mm_mol.get_zetas()[[cl]])
        dhdl_ewald[i] += cp.einsum('i,i->', mm_pot[0], mf.get_qm_charges(dm))
        dhdl_ewald[i] += cp.einsum('ix,ix->', mm_pot[1], mf.get_qm_dipoles(dm))
        dhdl_ewald[i] += cp.einsum('ixy,ixy->', mm_pot[2], mf.get_qm_quadrupoles(dm))
        dhdl_ewald[i] = dhdl_ewald[i].get()
    ## D4
    e_disp, g_disp, dhdl_disp = d4(mf.mol, pidx_in_mol, lambdas)
    e_qmmm += e_disp
    f_qmmm[qm_indexes] -= g_disp
    ## MM
    dhdl_mm = list()
    ew_eta, ew_cut = mm_mol.get_ewald_params()
    e_mm = 0
    for cl in clidx_in_mm:
        qcl = mm_mol.atom_charge(cl)
        # cl-mm ewald hess
        cl_mm_hess0, cl_mm_hess1 = get_ewald_hess(
            mm_mol,
            mm_mol.atom_coords()[[cl]],
            mm_mol.atom_coords())
        e_mm -= 0.5 * qcl * cl_mm_hess0[0,cl].get() * qcl
        e_mm += qcl * cl_mm_hess0[0].get() @ mm_mol.atom_charges()
        e_mm -= ew_eta / np.sqrt(np.pi) * qcl**2
        dhdl_mm.append(0)
        dhdl_mm[-1] -= 2 * (-1.0) * ew_eta / np.sqrt(np.pi) * qcl
        dhdl_mm[-1] += -1.0 * cl_mm_hess0[0].get() @ mm_mol.atom_charges()
        assert cp.linalg.norm(cl_mm_hess1[0,cl]) < 1e-9
        f_qmmm[mm_indexes] += qcl * np.einsum(
            'jx,j->jx', cl_mm_hess1[0].get(), mm_mol.atom_charges())
        f_qmmm[mm_indexes[cl]] -= qcl * np.einsum(
            'jx,j->x', cl_mm_hess1[0].get(), mm_mol.atom_charges())
    e_qmmm += e_mm

    dhdl = np.zeros_like(lambdas)
    dhdl += dhdl_elec
    dhdl += dhdl_nuc
    dhdl += dhdl_res
    dhdl += dhdl_ewald
    dhdl += dhdl_disp
    dhdl += dhdl_mm
    print("dhdl[0] =", dhdl[0])

    print("efv total time =", t3 - t0)
    return e_qmmm, f_qmmm, None, init_dict


if __name__ == "__main__":
    fp = open("./geom.xyz")
    fp.readline(); fp.readline()
    coords = np.array([line.split()[1:] for line in fp], dtype=float)
    fp.close()
    coords = coords * A / Bohr
    box = np.diag([47.66706371,   47.66706371,  47.66706371]) * A / Bohr
    e, f, v, d = efv_scan(coords, box, dict())
    print("e =", e)
