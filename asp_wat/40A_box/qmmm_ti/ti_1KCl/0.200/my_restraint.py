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
    h_pos = (mol.atom_coord(4) - mol.atom_coord(1)) / 1.5 + mol.atom_coord(6)
    Eres, gres = bond_energy_grad(mol.atom_coord(p)-h_pos, k_res, 0.0)
    dhdl_res = [Eres * (-1)]
    Eres *= (1-l)
    gres *= (1-l)
    g = np.zeros_like(mol.atom_coords())
    g[p] += gres
    g[4] -= gres / 1.5
    g[1] += gres / 1.5
    g[6] -= gres
    return Eres, g, dhdl_res
