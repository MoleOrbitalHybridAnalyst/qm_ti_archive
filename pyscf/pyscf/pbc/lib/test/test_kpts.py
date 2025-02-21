#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.scf import khf
from pyscf.pbc.lib import kpts_helper
from pyscf import lib

def setUpModule():
    global cell
    cell = pbcgto.Cell()
    cell.atom = 'He 0 0 0'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''
    cell.verbose = 0
    cell.space_group_symmetry = True
    cell.build()

def tearDownModule():
    global cell
    del cell

class KnownValues(unittest.TestCase):
    def test_kconserve(self):
        kpts = cell.make_kpts([3,4,5])
        kconserve = kpts_helper.get_kconserv(cell, kpts)
        self.assertAlmostEqual(lib.finger(kconserve), 84.88659638289468, 9)
        kpts = cell.make_kpts([3,4,5], space_group_symmetry=True)
        kconserve = kpts_helper.KptsHelper(cell, kpts).kconserv
        self.assertAlmostEqual(lib.finger(kconserve), 84.88659638289468, 9)

    def test_kconserve3(self):
        kpts = cell.make_kpts([2,2,2])
        nkpts = kpts.shape[0]
        kijkab = [range(nkpts),range(nkpts),1,range(nkpts),range(nkpts)]
        kconserve = kpts_helper.get_kconserv3(cell, kpts, kijkab)
        self.assertAlmostEqual(lib.finger(kconserve), -3.1172758206126852, 0)

    def test_conj_kpts(self):
        kpts = cell.make_kpts([8,5,2])
        idx = np.arange(kpts.shape[0])
        np.random.shuffle(idx)
        kpts = kpts[idx]
        idx = kpts_helper.conj_mapping(cell, kpts)
        check = (kpts+kpts[idx]).dot(cell.lattice_vectors().T/(2*np.pi)) + 1e-14
        self.assertAlmostEqual(np.modf(check)[0].max(), 0, 12)

    def test_symmetrize(self):
        pass

    def test_group_by_conj_paris(self):
        kpts = cell.make_kpts([3,4,1])
        nkpts = len(kpts)
        ukpts, _, uniq_inv = kpts_helper.unique_with_wrap_around(
            cell, (kpts[None,:,:] - kpts[:,None,:]).reshape(-1, 3))
        pairs = kpts_helper.group_by_conj_pairs(cell, ukpts)[0]
        self.assertEqual(
            pairs, [(0, 0), (2, 2), (1, 3), (4, 8), (5, 11), (6, 10), (7, 9)])
        for i, j in pairs:
            if j is not None:
                idx = np.where(uniq_inv == i)[0] // nkpts
                idy = np.where(uniq_inv == i)[0] % nkpts
                self.assertTrue(np.array_equiv(np.sort(idy*nkpts+idx), np.where(uniq_inv == j)[0]))

if __name__ == "__main__":
    print("Tests for kpts_helper")
    unittest.main()
