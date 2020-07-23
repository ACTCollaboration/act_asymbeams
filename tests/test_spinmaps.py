import numpy as np
import unittest

from pixell import enmap
from act_asymbeams.spinmaps import spin2eb
from act_asymbeams.spinmaps import eb2spin
from act_asymbeams.spinmaps import blm2bl
from act_asymbeams.spinmaps import compute_spinmap_real
from act_asymbeams.spinmaps import trunc_alm
from act_asymbeams.spinmaps import blm2eb

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

    def test_eb2spin(self):
        
        blmB = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmE = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blmm2_exp = -1 * (blmE - 1j * blmB)
        blmp2_exp = -1 * (blmE + 1j * blmB)

        blmm2, blmp2 = eb2spin(blmE, blmB)
        np.testing.assert_array_almost_equal(blmm2, blmm2_exp)
        np.testing.assert_array_almost_equal(blmp2, blmp2_exp)

    def test_eb2spin_inplace(self):

        blmB = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmE = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blmm2_exp = -1 * (blmE - 1j * blmB)
        blmp2_exp = -1 * (blmE + 1j * blmB)

        blmm2, blmp2 = eb2spin(blmE, blmB, inplace=True)

        np.testing.assert_array_almost_equal(blmm2, blmm2_exp)
        np.testing.assert_array_almost_equal(blmp2, blmp2_exp)

        self.assertTrue(np.shares_memory(blmE, blmp2))
        self.assertTrue(np.shares_memory(blmB, blmm2))

    def test_eb2spin_inplace_round(self):

        blmB_exp = np.array(
            [0, 0, 1j, 1j, 1j, 0, 2j, 2j, 2j, 3j, 3j, 3j, 4j, 4j, 5j],
            dtype=np.complex128)        
        blmE_exp = np.array([0, 0, 1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 4, 5],
                          dtype=np.complex128)
        
        blmm2, blmp2 = eb2spin(blmE_exp, blmB_exp, inplace=True)
        blmE, blmB = spin2eb(blmm2, blmp2, inplace=True)
        
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)

        self.assertTrue(np.shares_memory(blmE, blmE_exp))
        self.assertTrue(np.shares_memory(blmB, blmB_exp))

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
    
    def test_blm2eb(self):
        
        blm = np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j, 4j])         

        blmm2_exp = np.asarray([0, 0, -3j, -3j, 0, 2j, 2j, 1, 1, 2j])
        blmp2_exp = np.asarray([0, 0, 3j, 3j, 0, 0, 4j, 0, 0, 0])

        blmE_exp = -0.5 * (blmp2_exp + blmm2_exp)
        blmB_exp = 0.5j * (blmp2_exp - blmm2_exp)

        blmE, blmB = blm2eb(blm)
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)
        self.assertTrue(blmE.flags['OWNDATA'])
        self.assertTrue(blmB.flags['OWNDATA'])

    def test_blm2eb_mmax(self):
        
        blm = np.asarray([1, 1, 1, 1, 2j, 2j, 2j, 3j, 3j])         

        blmm2_exp = np.asarray([0, 0, -3j, -3j, 0, 2j, 2j, 1, 1])
        blmp2_exp = np.asarray([0, 0, 3j, 3j, 0, 0, 0, 0, 0])

        blmE_exp = -0.5 * (blmp2_exp + blmm2_exp)
        blmB_exp = 0.5j * (blmp2_exp - blmm2_exp)

        blmE, blmB = blm2eb(blm, mmax=2)
        np.testing.assert_array_almost_equal(blmE, blmE_exp)
        np.testing.assert_array_almost_equal(blmB, blmB_exp)
        self.assertTrue(blmE.flags['OWNDATA'])
        self.assertTrue(blmB.flags['OWNDATA'])
        
    def test_blm2eb_err(self):
        
        blm = np.asarray([1, 1, 1, 1, 2j, 2j, 2j])         
        self.assertRaises(ValueError, blm2eb, blm, **{'mmax' : 1})

        blm = np.ones((2, 10))
        self.assertRaises(ValueError, blm2eb, blm)
