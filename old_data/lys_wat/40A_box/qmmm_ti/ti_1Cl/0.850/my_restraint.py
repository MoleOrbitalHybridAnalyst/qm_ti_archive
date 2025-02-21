import numpy as np

k_res = 0.1 # hartree / Bohr^2

def bond_energy_grad(r, k, a):
    rnorm = np.linalg.norm(r)
    E = k * (rnorm - a)**2
    g = 2 * k * (rnorm - a) / rnorm * r
    return E, g

def compute_restraint(mol, pidx_in_mol, lambdas):
    assert len(pidx_in_mol) == 1
    assert len(lambdas) == 1
    p = pidx_in_mol[0]
    l = lambdas[0]
    # obtained with ../../../qmmm_equil/nvt_1Cl/regression.py on the first 5 ps
    x = [-0.77047501,  3.77812947, -1.00382723, -1.00382723]
    h_pos  = mol.atom_coord(10) * x[0]
    h_pos += mol.atom_coord(13) * x[1]
    h_pos += mol.atom_coord(14) * x[2]
    h_pos += mol.atom_coord(16) * x[3]
    Eres, gres = bond_energy_grad(mol.atom_coord(p)-h_pos, k_res, 0.0)
    dhdl_res = [Eres * (-1)]
    Eres *= (1-l)
    gres *= (1-l)
    g = np.zeros_like(mol.atom_coords())
    g[p] += gres
    g[10] -= gres * x[0]
    g[13] -= gres * x[1]
    g[14] -= gres * x[2]
    g[16] -= gres * x[3]
    return Eres, g, dhdl_res
