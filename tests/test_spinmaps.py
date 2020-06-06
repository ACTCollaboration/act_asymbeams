import numpy as np
import unittest

from pixell import enmap
from act_asymbeams.spinmaps import spin2eb, blm2bl, compute_spinmap_real, trunc_alm

class TestSpinMap(unittest.TestCase):

    def test_spin2eb(self):
        
        blmm2 = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmp2 = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blmE_exp = -0.5 * (blmp2 + blmm2)
        blmB_exp = 0.5j * (blmp2 - blmm2)

        blmE, blmB = spin2eb(blmm2, blmp2)
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)

    def test_spin2eb_spin(self):
        
        spin = 1
        blmm2 = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmp2 = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blmE_exp = -0.5 * (blmp2 + (-1) ** spin * blmm2)
        blmB_exp = 0.5j * (blmp2 - (-1) ** spin * blmm2)

        blmE, blmB = spin2eb(blmm2, blmp2, spin=spin)
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)

    def test_spin2eb_inplace(self):
        
        blmm2 = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmp2 = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)

        blmE_exp = -0.5 * (blmp2 + blmm2)
        blmB_exp = 0.5j * (blmp2 - blmm2)

        blmE, blmB = spin2eb(blmm2, blmp2, inplace=True, batchsize=5)
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)
        self.assertTrue(np.shares_memory(blmp2, blmE))
        self.assertTrue(np.shares_memory(blmm2, blmB))

    def test_blm2bl(self):
        
        blm = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        

        bell = blm2bl(blm, m=1, norm=False)
        bell_exp = np.array([0, 2j, 2j, 2j], dtype=np.complex128)        
        
        np.testing.assert_almost_equal(bell, bell_exp)
        self.assertFalse(np.shares_memory(blm, bell))

    def test_blm2bl_inplace(self):
        
        blm = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        

        bell = blm2bl(blm, m=1, norm=False, copy=False)
        bell_exp = np.array([0, 2j, 2j, 2j], dtype=np.complex128)        
        
        np.testing.assert_almost_equal(bell, bell_exp)
        self.assertTrue(np.shares_memory(blm, bell))

    def test_blm2bl_norm(self):
        
        blm = np.array(
            [10, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        

        bell = blm2bl(blm, m=1, norm=True)
        ell = np.arange(1, 5, dtype=float)
        qell = np.sqrt(4 * np.pi / (2 * ell + 1))
        bell_exp = np.array([0, 2j, 2j, 2j], dtype=np.complex128) * qell
        bell_exp /= (blm[0] * np.sqrt(4 * np.pi))
        np.testing.assert_almost_equal(bell, bell_exp)
        self.assertFalse(np.shares_memory(blm, bell))        

    def test_compute_spinmap_real(self):

        blm = np.array(
            [10, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blm_mmax = 4
        alm = np.zeros_like(blm)
        alm[0] = 1
        spin = 0
        shape = (101, 200)
        res = [np.pi / float(shape[0] - 1), 2*np.pi / float(shape[1])]

        shape, wcs = enmap.fullsky_geometry(res=res, shape=shape)
        omap = enmap.zeros(shape, wcs)

        compute_spinmap_real(alm, blm, blm_mmax, spin, omap)

        omap_exp = enmap.full(shape, wcs, 1/np.sqrt(4 * np.pi))
        np.testing.assert_almost_equal(omap, omap_exp)

    def test_trunc_alm(self):

        alm = np.array(
            [10, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        

        alm_trunc = trunc_alm(alm, 3)        
        alm_exp = np.array([10, 0, 1j, 1j, 0, 2j, 2j, 3j, 3j, 4j],
            dtype=np.complex128)        
        
        np.testing.assert_almost_equal(alm_trunc, alm_exp)
        self.assertFalse(alm_trunc.flags['OWNDATA'])
        self.assertTrue(alm_trunc.base is alm)

    
