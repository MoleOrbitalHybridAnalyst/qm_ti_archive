import numpy
import numpy as np
from pyscf import gto
from pyscf import pbc
from pyscf import qmmm
from pyscf.gto.mole import is_au
from pyscf.data.elements import charge
from pyscf.lib import param, logger
from pyscf.pbc.gto.cell import _cut_mesh_for_ewald
from scipy.special import erf, erfc, lambertw
from pyscf import lib

class Cell(qmmm.mm_mole.Mole, pbc.gto.Cell):
    '''Cell class for MM particles.

    Args:
        atoms : geometry of MM particles (unit Bohr).

            | [[atom1, (x, y, z)],
            |  [atom2, (x, y, z)],
            |  ...
            |  [atomN, (x, y, z)]]

    Kwargs:
        charges : 1D array
            fractional charges of MM particles
        zeta : 1D array
            Gaussian charge distribution parameter.
            rho(r) = charge * Norm * exp(-zeta * r^2)

    '''
    def __init__(self, atoms, a, 
            rcut_ewald=None, rcut_hcore=None,
            charges=None, zeta=None):
        pbc.gto.Cell.__init__(self)
        self.atom = self._atom = atoms
        self.unit = 'Bohr'
        self.charge_model = 'point'
        assert np.linalg.norm(a - np.diag(np.diag(a))) < 1e-12
        self.a = a
        if rcut_ewald is None:
            rcut_ewald = min(np.diag(a)) * 0.9
            logger.warn(self, "Setting rcut_ewaldto be 0.9 * box size")
        if rcut_hcore is None:
            rcut_hcore = np.linalg.norm(np.diag(a)) / 2
            logger.warn(self, "Setting rcut_hcore to be half box size")
        assert rcut_ewald < min(np.diag(a)), "Only rcut_ewald < box size implemented"
        self.rcut_ewald = rcut_ewald
        self.rcut_hcore = rcut_hcore

        # Initialize ._atm and ._env to save the coordinates and charges and
        # other info of MM particles
        natm = len(atoms)
        _atm = numpy.zeros((natm,6), dtype=numpy.int32)
        _atm[:,gto.CHARGE_OF] = [charge(a[0]) for a in atoms]
        coords = numpy.asarray([a[1] for a in atoms], dtype=numpy.double)
        if charges is None:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_POINT
            charges = _atm[:,gto.CHARGE_OF:gto.CHARGE_OF+1]
        else:
            _atm[:,gto.NUC_MOD_OF] = gto.NUC_FRAC_CHARGE
            charges = numpy.asarray(charges)[:,numpy.newaxis]

        self._env = numpy.append(numpy.zeros(gto.PTR_ENV_START),
                                 numpy.hstack((coords, charges)).ravel())
        _atm[:,gto.PTR_COORD] = gto.PTR_ENV_START + numpy.arange(natm) * 4
        _atm[:,gto.PTR_FRAC_CHARGE] = gto.PTR_ENV_START + numpy.arange(natm) * 4 + 3

        if zeta is not None:
            self.charge_model = 'gaussian'
            zeta = numpy.asarray(zeta, dtype=float).ravel()
            self._env = numpy.append(self._env, zeta)
            _atm[:,gto.PTR_ZETA] = gto.PTR_ENV_START + natm*4 + numpy.arange(natm)

        self._atm = _atm

        eta, _ = self.get_ewald_params()
        e = self.precision
        Q = np.sum(self.atom_charges()**2)
        L = self.vol**(1/3)
        kmax = np.sqrt(3)*eta/2/np.pi * np.sqrt(lambertw( 4*Q**(2/3)/3/np.pi**(2/3)/L**2/eta**(2/3) / e**(4/3) ).real)
        self.mesh = np.ceil(np.diag(self.lattice_vectors()) * kmax).astype(int) * 2 + 1

        self._built = True

    def get_lattice_Ls(self):
        Ts = lib.cartesian_prod((np.arange(-1, 2),
                                 np.arange(-1, 2),
                                 np.arange(-1, 2)))
        Lall = np.dot(Ts, self.lattice_vectors())
        return Lall

    def get_ewald_params(self, precision=None, rcut=None):
        if rcut is None:
            ew_cut = self.rcut_ewald
        else:
            ew_cut = rcut
        if precision is None:
            precision = self.precision
        e = precision
        Q = np.sum(self.atom_charges()**2)
        ew_eta = 1 / ew_cut * np.sqrt(lambertw(1/e*np.sqrt(Q/2/self.vol)).real)
        return ew_eta, ew_cut

    def get_ewald_pot(self, coords1, coords2=None, charges2=None):
        assert self.dimension == 3
        assert (coords2 is None and charges2 is None) or (coords2 is not None and charges2 is not None)

        if charges2 is not None:
            assert len(charges2) == len(coords2)
        else:
            coords2 = coords1

        ew_eta, ew_cut = self.get_ewald_params()
        mesh = self.mesh
        
        logger.debug(self, f"Ewald exponent {ew_eta}")

        # TODO Lall should respect ew_rcut
        Lall = self.get_lattice_Ls()

        all_coords2 = lib.direct_sum('jx-Lx->Ljx', coords2, Lall).reshape(-1,3)
        if charges2 is not None:
            all_charges2 = np.hstack([charges2] * len(Lall))
        else:
            all_charges2 = None
        dist2 = lib.direct_sum('jx-x->jx', all_coords2, np.mean(coords1, axis=0))
        dist2 = lib.einsum('jx,jx->j', dist2, dist2)
        R = lib.direct_sum('ix-jx->ijx', coords1, all_coords2)
        r = np.sqrt(lib.einsum('ijx,ijx->ij', R, R))
        r[r<1e-16] = 1e100
        rmax_qm = max(np.linalg.norm(coords1 - np.mean(coords1, axis=0), axis=-1))

        # substract the real-space Coulomb within rcut_hcore
        mask = dist2 <= self.rcut_hcore**2
        Tij = 1 / r[:,mask]
        Rij = R[:,mask]
        Tija = -lib.einsum('ijx,ij->ijx', Rij, Tij**3)
        Tijab  = 3 * lib.einsum('ija,ijb->ijab', Rij, Rij) 
        Tijab  = lib.einsum('ijab,ij->ijab', Tijab, Tij**5)
        Tijab -= lib.einsum('ij,ab->ijab', Tij**3, np.eye(3))
        if all_charges2 is not None:
            charges = all_charges2[mask]
            # ew0 = -d^2 E / dQi dqj qj
            # ew1 = -d^2 E / dDia dqj qj
            # ew2 = -d^2 E / dOiab dqj qj
            # qm pc - mm pc
            ewovrl0 = -lib.einsum('ij,j->i', Tij, charges)
            # qm dip - mm pc
            ewovrl1 = -lib.einsum('j,ija->ia', charges, Tija)
            # qm quad - mm pc
            ewovrl2 = -lib.einsum('j,ijab->iab', charges, Tijab) / 3
        else:
            # FIXME a too small rcut_hcore truncates QM atoms, while this correction
            # should be applied to all QM pairs regardless of rcut_hcore
            assert r[:,mask].shape[0] == r[:,mask].shape[1]   # real-space should not see qm images
            # ew00 = -d^2 E / dQi dQj
            # ew01 = -d^2 E / dQi dDja
            # ew11 = -d^2 E / dDia dDjb
            # ew02 = -d^2 E / dQi dOjab
            ewovrl00 = -Tij
            ewovrl01 =  Tija
            ewovrl11 =  Tijab
            ewovrl02 = -Tijab / 3

        # difference between MM gaussain charges and MM point charges
        if all_charges2 is not None and self.charge_model == 'gaussian':
            mask = dist2 > self.rcut_hcore**2
            min_expnt = min(self.get_zetas())
            max_ewrcut = pbc.gto.cell._estimate_rcut(min_expnt, 0, 1., self.precision)
            cut2 = (max_ewrcut + rmax_qm)**2
            mask = mask & (dist2 <= cut2)
            expnts = np.hstack([np.sqrt(self.get_zetas())] * len(Lall))[mask]
            r_ = r[:,mask]
            R_ = R[:,mask]
            ekR = np.exp(-lib.einsum('j,ij->ij', expnts**2, r_**2))
            Tij = erfc(lib.einsum('j,ij->ij', expnts, r_)) / r_
            invr3 = (Tij + lib.einsum('j,ij->ij', expnts, 2/np.sqrt(np.pi)*ekR)) / r_**2
            Tija = -lib.einsum('ijx,ij->ijx', R_, invr3)
            Tijab  = 3 * lib.einsum('ija,ijb,ij->ijab', R_, R_, 1/r_**2)
            Tijab -= lib.einsum('ij,ab->ijab', np.ones_like(r_), np.eye(3))
            invr5 = invr3 + lib.einsum('j,ij->ij', expnts**3, 4/3/np.sqrt(np.pi) * ekR)
            Tijab = lib.einsum('ijab,ij->ijab', Tijab, invr5)
            Tijab += lib.einsum('j,ij,ab->ijab', expnts**3, 4/3/np.sqrt(np.pi)*ekR, np.eye(3))
            ewovrl0 -= lib.einsum('ij,j->i', Tij, all_charges2[mask])
            ewovrl1 -= lib.einsum('j,ija->ia', all_charges2[mask], Tija)
            ewovrl2 -= lib.einsum('j,ijab->iab', all_charges2[mask], Tijab) / 3

        # ewald real-space sum
        cut2 = (ew_cut + rmax_qm)**2
        mask = dist2 <= cut2
        r = r[:,mask]
        R = R[:,mask]
        if all_charges2 is not None:
            all_charges2 = all_charges2[mask]
        else:
            if r.shape[0] != r.shape[1]:
                raise NotImplementedError("QM image cannot be within ewald cutoff of QM")
        ekR = np.exp(-ew_eta**2 * r**2)
        # Tij = \hat{1/r} = f0 / r = erfc(r) / r
        Tij = erfc(ew_eta * r) / r
        # Tija = -Rija \hat{1/r^3} = -Rija / r^2 ( \hat{1/r} + 2 eta/sqrt(pi) exp(-eta^2 r^2) )
        invr3 = (Tij + 2*ew_eta/np.sqrt(np.pi) * ekR) / r**2
        Tija = -lib.einsum('ijx,ij->ijx', R, invr3)
        # Tijab = (3 RijaRijb - Rij^2 delta_ab) \hat{1/r^5}
        Tijab  = 3 * lib.einsum('ija,ijb,ij->ijab', R, R, 1/r**2)
        Tijab -= lib.einsum('ij,ab->ijab', np.ones_like(r), np.eye(3))
        invr5 = invr3 + 4/3*ew_eta**3/np.sqrt(np.pi) * ekR # NOTE this is invr5 * r**2
        Tijab = lib.einsum('ijab,ij->ijab', Tijab, invr5)
        # NOTE the below is present in Eq 8 but missing in Eq 12
        Tijab += 4/3*ew_eta**3/np.sqrt(np.pi)*lib.einsum('ij,ab->ijab', ekR, np.eye(3))

        if all_charges2 is not None:
            ewovrl0 += lib.einsum('ij,j->i', Tij, all_charges2)
            ewovrl1 += lib.einsum('j,ija->ia', all_charges2, Tija)
            ewovrl2 += lib.einsum('j,ijab->iab', all_charges2, Tijab) / 3
        else:
            ewovrl00 += Tij
            ewovrl01 -= Tija
            ewovrl11 -= Tijab
            ewovrl02 += Tijab / 3
        ekR = Tij = invr3 = Tijab = invr5 = None

        if all_charges2 is not None:
            pass
        else:
            ewself01 = 0
            ewself02 = 0
            # -d^2 Eself / dQi dQj
            ewself00 = -np.eye(len(coords1)) * 2 * ew_eta / np.sqrt(np.pi)
            # -d^2 Eself / dDia dDjb
            ewself11 = -lib.einsum('ij,ab->ijab', np.eye(len(coords1)), np.eye(3)) \
                    * 4 * ew_eta**3 / 3 / np.sqrt(np.pi)

        R = r = dist2 = all_charges2 = mask = None

        # g-space sum (using g grid)
        logger.debug(self, f"Ewald mesh {mesh}")

        Gv, Gvbase, weights = self.get_Gv_weights(mesh)
        absG2 = lib.einsum('gx,gx->g', Gv, Gv)
        absG2[absG2==0] = 1e200

        coulG = 4*np.pi / absG2
        coulG *= weights
        # NOTE Gpref is actually Gpref*2
        Gpref = np.exp(-absG2/(4*ew_eta**2)) * coulG

        GvR2 = lib.einsum('gx,ix->ig', Gv, coords2)
        cosGvR2 = np.cos(GvR2)
        sinGvR2 = np.sin(GvR2)

        if charges2 is not None:
            GvR1 = lib.einsum('gx,ix->ig', Gv, coords1)
            cosGvR1 = np.cos(GvR1)
            sinGvR1 = np.sin(GvR1)
            zcosGvR2 = lib.einsum("i,ig->g", charges2, cosGvR2)
            zsinGvR2 = lib.einsum("i,ig->g", charges2, sinGvR2)
            # qm pc - mm pc
            ewg0  = lib.einsum('ig,g,g->i', cosGvR1, zcosGvR2, Gpref)
            ewg0 += lib.einsum('ig,g,g->i', sinGvR1, zsinGvR2, Gpref)
            # qm dip - mm pc
            ewg1  = lib.einsum('gx,ig,g,g->ix', Gv, cosGvR1, zsinGvR2, Gpref)
            ewg1 -= lib.einsum('gx,ig,g,g->ix', Gv, sinGvR1, zcosGvR2, Gpref)
            # qm quad - mm pc
            ewg2  = -lib.einsum('gx,gy,ig,g,g->ixy', Gv, Gv, cosGvR1, zcosGvR2, Gpref)
            ewg2 += -lib.einsum('gx,gy,ig,g,g->ixy', Gv, Gv, sinGvR1, zsinGvR2, Gpref)
            ewg2 /= 3
        else:
            # qm pc - qm pc
            ewg00  = lib.einsum('ig,jg,g->ij', cosGvR2, cosGvR2, Gpref)
            ewg00 += lib.einsum('ig,jg,g->ij', sinGvR2, sinGvR2, Gpref)
            # qm pc - qm dip
            ewg01  = lib.einsum('gx,ig,jg,g->ijx', Gv, sinGvR2, cosGvR2, Gpref)
            ewg01 -= lib.einsum('gx,ig,jg,g->ijx', Gv, cosGvR2, sinGvR2, Gpref)
            # qm dip - qm dip
            ewg11  = lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, cosGvR2, cosGvR2, Gpref)
            ewg11 += lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, sinGvR2, sinGvR2, Gpref)
            # qm pc - qm quad
            ewg02  = -lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, cosGvR2, cosGvR2, Gpref)
            ewg02 += -lib.einsum('gx,gy,ig,jg,g->ijxy', Gv, Gv, sinGvR2, sinGvR2, Gpref)
            ewg02 /= 3

        if charges2 is not None:
            return ewovrl0 + ewg0, ewovrl1 + ewg1, ewovrl2 + ewg2
        else:
            return ewovrl00 + ewself00 + ewg00, \
                   ewovrl01 + ewself01 + ewg01, \
                   ewovrl11 + ewself11 + ewg11, \
                   ewovrl02 + ewself02 + ewg02

def create_mm_mol(atoms_or_coords, a, charges=None, radii=None, 
        rcut_ewald=None, rcut_hcore=None, unit='Angstrom'):
    '''Create an MM object based on the given coordinates and charges of MM
    particles.

    Args:
        atoms_or_coords : array-like
            Cartesian coordinates of MM atoms, in the form of a 2D array:
            [(x1, y1, z1), (x2, y2, z2), ...]
        a : (3,3) ndarray
            Lattice primitive vectors. Each row represents a lattice vector
            Reciprocal lattice vectors are given by  b1,b2,b3 = 2 pi inv(a).T

    Kwargs:
        charges : 1D array
            The charges of MM atoms.
        radii : 1D array
            The Gaussian charge distribuction radii of MM atoms.
        unit : string
            The unit of the input. Default is 'Angstrom'.
    '''
    if isinstance(atoms_or_coords, numpy.ndarray):
        # atoms_or_coords == np.array([(xx, xx, xx)])
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    elif (isinstance(atoms_or_coords, (list, tuple)) and
          atoms_or_coords and
          isinstance(atoms_or_coords[0][1], (int, float))):
        # atoms_or_coords == [(xx, xx, xx)]
        # Patch ghost atoms
        atoms = [(0, c) for c in atoms_or_coords]
    else:
        atoms = atoms_or_coords
    atoms = gto.format_atom(atoms, unit=unit)

    if radii is None:
        zeta = None
    else:
        radii = numpy.asarray(radii, dtype=float).ravel()
        if not is_au(unit):
            radii = radii / param.BOHR
        zeta = 1 / radii**2

    kwargs = {'charges': charges, 'zeta': zeta}

    if not is_au(unit):
        a = a / param.BOHR
        if rcut_ewald is not None:
            rcut_ewald = rcut_ewald / param.BOHR
        if rcut_hcore is not None:
            rcut_hcore = rcut_hcore / param.BOHR

    if rcut_ewald is not None:
        kwargs['rcut_ewald'] = rcut_ewald
    if rcut_hcore is not None:
        kwargs['rcut_hcore'] = rcut_hcore

    return Cell(atoms, a, **kwargs)

create_mm_cell = create_mm_mol
