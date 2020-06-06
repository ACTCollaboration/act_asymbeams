import numpy as np
import os

import healpy as hp
from pixell import enmap, curvedsky

def trunc_alm(alm, lmax_new, mmax_old=None, return_contiguous=True):
    '''
    Inplace truncation of alm array to new lmax.

    Parameters
    ----------
    alm : (nelem) or (..., nelem) array
        Healpix ordered alm array.
    lmax : int
        Truncate to this lmax.
    mmax_old : int, optional
        Maximum m-mode of alm array.
    return_contiguous : bool, optional
        Return c-contiguous array (only matters for 2d input).

    Returns
    -------
    alm : array
        Truncated array.

    Raises
    ------
    ValueError
        If new lmax exceeds old lmax
    
    Notes
    -----
    Returns view into input array. Up to user to reallocate
    memory if so desired.
    '''

    ndim =  alm.ndim
    alm = np.atleast_2d(alm)

    lmax = hp.Alm.getlmax(alm.shape[-1], mmax=mmax_old)
    if lmax < lmax_new:
        raise ValueError('new lmax should be smaller than old lmax')
    if mmax_old is None:
        mmax_old = lmax
    elldiff = lmax - lmax_new

    for m in range(min(lmax_new, mmax_old) + 1):

        start_old = hp.Alm.getidx(lmax, m, m)
        end_old = start_old + lmax - m + 1

        start_new = hp.Alm.getidx(lmax_new, m, m)
        end_new = start_new + lmax_new - m + 1

        alm[...,start_new:end_new] = alm[...,start_old:end_old-elldiff]

    alm = alm[...,:hp.Alm.getsize(lmax_new, mmax=min(mmax_old, lmax_new))]
    alm = alm[0] if ndim == 1 else alm
    alm = np.ascontiguousarray(alm) if ndim > 1 and return_contiguous else alm

    return alm

def eb2spin(almE, almB):
    '''
    Convert to E and B mode coefficients
    to spin-harmonic coefficients.

    Arguments
    ---------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes

    Returns
    -------
    almm2 : array-like
       Healpix-ordered complex array with spin-(-2)
       coefficients
    almp2 : array-like
       Healpix-ordered complex array with spin-(+2)
       coefficients
    '''

    almm2 = -1 * (almE - 1j * almB)
    almp2 = -1 * (almE + 1j * almB)

    return almm2, almp2

def spin2eb(almm2, almp2, spin=2, inplace=False, batchsize=10000):
    '''
    Convert spin-harmonic coefficients
    to E and B mode coefficients.

    Paramters
    ---------
    almm2 : array-like
       Healpix-ordered complex array with spin-(-2)
       coefficients
    almp2 : array-like
       Healpix-ordered complex array with spin-(+2)
       coefficients
    spin : int, optional
        Spin of input. Odd spins receive relative
        minus sign between input in order to be consistent
        with HEALPix alm2map_spin.
    inplace : bool, optional
        Use almp2 and almm2 input arrays for almE and almB.
    batchsize : int, optional
        Calculation is done in steps of this size.

    Returns
    -------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes\

    Raises
    ------
    ValueError
        If spin is not an integer.

    Notes
    -----
    See https://healpix.jpl.nasa.gov/html/subroutinesnode12.htm
    '''

    if int(spin) != spin:
        raise ValueError('Spin must be integer')

    if not inplace:
        almm2 = almm2.copy()
        almp2 = almp2.copy()

    nelem = almm2.size

    for start in range(0, nelem, batchsize):
        end = min(start + batchsize, nelem)

        almm2_temp = almm2[start:end].copy()
        almp2_temp = almp2[start:end].copy()

        # Reuse arrays: p2 -> E, m2 -> B.
        almp2[start:end] = almp2_temp + almm2_temp * (-1.) ** spin
        almp2[start:end] /= -2.

        almm2[start:end] = almp2_temp - almm2_temp * (-1.) ** spin
        almm2[start:end] *= (1j / 2.)

    almE, almB = almp2, almm2

    return almE, almB

def blm2bl(blm, m=0, mmax=None, copy=True, full=False, norm=True):
    '''
    A tool to return blm for a fixed m-mode

    Parameters
    ----------
    blm : array
        Spherical harmonics of the beam
    m : int, optional
        The m-mode being requested (note m >= 0)
    mmax : int, None, optional
        Maximum m in blm.
    copy : bool, optional
        Return copied slice or not.
    full : bool. optional
        If set, always return full-sized (lmax + 1) array
        Note, this always produces a copy.
    norm : bool, optional
        Normalize bell: 
           
        bl_norm = sqrt(4pi/(2l+1)) * bl / (b00 * sqrt(4pi))
    
        Note always produces copy.

    Returns
    -------
    bl : array
        Array of bl's for the m-mode requested
    '''

    if blm.ndim > 1:
        raise ValueError('blm should have have ndim == 1')
    if m < 0:
        raise ValueError('m cannot be negative')

    lmax = hp.Alm.getlmax(blm.size, mmax=mmax)

    start = hp.Alm.getidx(lmax, m, m)
    end = start + lmax + 1 - m

    bell = blm[start:end]

    if full:
        bell_full = np.zeros(lmax + 1, dtype=blm.dtype)
        bell_full[m:] = bell
        bell = bell_full

    if norm:
        lmin = 0 if full else m
        ells = np.arange(lmin, lmax + 1)
        q_ell = np.sqrt(4 * np.pi / (2 * ells + 1))

        bell = bell * q_ell
        bell /= blm[0] * np.sqrt(4 * np.pi)

    if copy and not (full or norm):
        return bell.copy()
    else:
        return bell

def compute_spinmap_real(alm, blm, blm_mmax, spin, omap):
    '''
    Return spinmap for a "real-valued" spin field with
    spin-weighted spherical harmonic coefficients given
    by sflm = alm * bls:

    spinmap_s = sum_lm sflm sYlm

    Parameters
    ----------
    alm : complex array
        Sky alms.
    blm : complex array
        Beam alms.
    blm_mmax : int
        Maximum m of blm array.
    spin : int
        Non-negative spin value.
    omap : enmap
        Real amd complex part of spinmap, shape is (2, ny, nx) if
        spin is nonzero, otherwise (1, ny, nx).

    Raises
    ------
    ValueError
        If spin is negative.
        If spin is higher than mmax.
    '''

    if spin < 0:
        raise ValueError('Negative spin provided.')
    if spin > blm_mmax:
        raise ValueError('Spin exceeds mmax beam.')

    lmax_alm = hp.Alm.getlmax(alm.size)
    lmax_blm = hp.Alm.getlmax(blm.size, mmax=blm_mmax)

    bell = blm2bl(blm, m=spin, mmax=blm_mmax, full=True, norm=True)

    if lmax_alm >= lmax_blm:
        lmax = lmax_blm
    elif lmax_alm < lmax_blm:
        lmax = lmax_alm
        bell = bell[:lmax_alm+1]

    if spin == 0:
        sflm = np.zeros((1, alm.size), dtype=alm.dtype)
    else:
        sflm = np.zeros((2, alm.size), dtype=alm.dtype)

    # The positive s values.
    sflm[0] = alm
    hp.almxfl(sflm[0], bell, inplace=True)

    if spin != 0:
        sflm[1] = alm
        # The negative s values: alm bl-s = alm bls^* (-1)^s.
        hp.almxfl(sflm[1], (-1) ** spin * np.conj(bell), inplace=True)

        # Turn into plus and minus (think E and B) modes for libsharp.
        spin2eb(sflm[1], sflm[0], spin=spin, inplace=True) 

    curvedsky.alm2map(sflm, omap, spin=spin)

    return omap
