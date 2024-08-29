from pyscf import scf, gto, mp, lib
import numpy as np

k_res = 0.01 # hartree/bohr^2
a_res = 0    # bohr

fp = open('./simulation.pos_0.xyz')
for iline, line in enumerate(fp):
    if iline % 6 == 0:
        atom = list()
    if iline % 6 > 1:
        atom.append(line)
    if iline % 6 == 5:
        pos_gh = np.array(atom[-1].split()[1:], dtype=float) / lib.param.BOHR
        mol = gto.Mole()
        mol.atom = atom[:-1]
        mol.basis = 'cc-pvdz'
        mol.verbose = 0
        mol.build()
        mf = scf.RHF(mol)
        Ehf = mf.kernel()
        mcc = mp.MP2(mf)
        Ecorr = mcc.kernel()[0]
        # restraint
        rr = mol.atom_coords()[:,None,:] - mol.atom_coords()[None]
        for i in range(mol.natm):
            rr[i, i] += 1e100
        def bond_energy_grad(r, k, a):
            rnorm = np.linalg.norm(r)
            E = k * (rnorm - a)**2
            g = 2 * k * (rnorm - a) / rnorm * r
            return E, g
        h_pos = (rr[0,1] + rr[0,2]) / 1.2 + mol.atom_coord(0)
        Eres, gres = bond_energy_grad(pos_gh-h_pos, k_res, a_res)
        print(Ehf + Ecorr + Eres)
        print(Ehf , Ecorr , Eres)
fp.close()
