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

def eb2spin(almE, almB, inplace=False, batchsize=10000):
    '''
    Convert to E and B mode coefficients
    to spin-harmonic coefficients.

    Arguments
    ---------
    almE : array-like
        Healpix ordered array with E-modes
    almB : array-like
        Healpix ordered array with B-modes
    inplace : bool, optional
        Use almE and almB input arrays for almp2 and almm2.
    batchsize : int, optional
        Calculation is done in steps of this size.

    Returns
    -------
    almm2 : array-like
       Healpix-ordered complex array with spin-(-2)
       coefficients
    almp2 : array-like
       Healpix-ordered complex array with spin-(+2)
       coefficients
    '''

    if not inplace:
        almE = almE.copy()
        almB = almB.copy()

    nelem = almE.size

    for start in range(0, nelem, batchsize):
        end = min(start + batchsize, nelem)

        almB_temp = almB[start:end].copy()
        almE_temp = almE[start:end].copy()

        almE[start:end] = almE_temp + 1j * almB_temp
        almE[start:end] *= -1.

        almB[start:end] = almE_temp - 1j * almB_temp
        almB[start:end] *= -1.

    almp2, almm2 = almE, almB

    return almm2, almp2

def spin2eb(almm2, almp2, spin=2, inplace=False, batchsize=10000):
    '''
    Convert spin-harmonic coefficients
    to E and B mode coefficients.

    Parameters
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
        Healpix ordered array with B-modes

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

def blm2bl(blm, m=0, mmax=None, copy=True, full=False, norm=True, b00=None):
    '''
    A tool to return blm for a fixed m-mode

    Parameters
    ----------
    blm : (nelem) array
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
    b00 : float, optional
        Use this value as b00 in normalization (used for spin != 0 blms)

    Returns
    -------
    bl : array
        Array of bl's for the m-mode requested
    '''

    if blm.ndim > 1:
        raise ValueError('blm should have have ndim == 1')
    if m < 0:
        raise ValueError('m cannot be negative')

    if b00 is None:
        b00 = blm[0]

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
        bell /= b00 * np.sqrt(4 * np.pi)

    if copy and not (full or norm):
        return bell.copy()
    else:
        return bell

def blm2eb(blm, mmax=None):
    '''
    Use the co-polar approximation to convert I blm into E and B blms.

    Parameters
    ----------
    blm : (nelem) array
        Spherical harmonics of the I beam.
    mmax : int, None, optional
        Maximum m in blm.

    Returns
    -------
    blmE : (nelem) array
        E-mode beam harmonic modes.
    blmB : (nelem) array
        B-mode beam harmonic modes.
    '''

    if blm.ndim > 1:
        raise ValueError('blm should have have ndim == 1.')

    lmax = hp.Alm.getlmax(blm.size, mmax=mmax)

    if mmax is None:
        mmax = lmax

    if mmax < 2:
        raise ValueError('mmax blm should be at least 2.')

    blmm2 = np.zeros_like(blm)
    blmp2 = np.zeros_like(blm)

    # Loop over m's in new arrays.
    for m in range(mmax + 1):

        # Slice into new blms.
        start = hp.Alm.getidx(lmax, m, m)
        end = start + lmax + 1 - m

        # First fill +2 blm.
        m_old = m + 2
        if abs(m_old) <= mmax:

            bell_old = blm2bl(blm, m=abs(m_old), mmax=mmax, full=True, norm=False)
            if m_old < 0:
                bell_old = np.conj(bell_old) * (-1) ** m_old

            # Length bell_old is always (lmax + 1).
            blmp2[start:end] = bell_old[m:]

        # Then fill -2 blm.
        m_old = m - 2
        if abs(m_old) <= mmax:

            bell_old = blm2bl(blm, m=abs(m_old), mmax=mmax, full=True, norm=False)
            if m_old < 0:
                bell_old = np.conj(bell_old) * (-1) ** m_old

            # Length bell_old is always (lmax + 1).
            blmm2[start:end] = bell_old[m:]

    # Since pm2 blm corresponds to a spin-pm2 field we should set m=1 elements to zero.
    # We need to do this manually for the m=ell=1 element, the others are zero already.
    blmm2[hp.Alm.getidx(lmax, 1, 1)] = 0
    blmp2[hp.Alm.getidx(lmax, 1, 1)] = 0
    
    return spin2eb(blmm2, blmp2, inplace=True)

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
        Real and complex part of spinmap, shape is (2, ny, nx) if
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

def compute_spinmap_complex(almE, almB, blmE, blmB, blm_mmax, spin, omap, b00=None):
    '''
    Return spinmap for a "complex-valued" spin field with
    spin-weighted spherical harmonic coefficients given
    by sflm = 2alm * -2bls:

    spinmap_s = sum_lm sflm sYlm

    Parameters
    ----------
    almE : complex array
        E-mode sky alms.
    almB : complex array
        B-mode sky alms.
    blmE : complex array
        E-mode beam alms.
    blmB : complex array
        B-mode beam alms.
    blm_mmax : int
        Maximum m of blm array.
    spin : int
        Spin value.
    omap : (2, ny, nx) enmap
        Real and complex part of spinmap.
    b00 : optional
        Use this value for b00 in normalization of beam.

    Raises
    ------
    ValueError
        If spin is higher than mmax.
        If E and B coefficients have different sizes.
    '''

    if spin > blm_mmax:
        raise ValueError('Spin exceeds mmax beam.')

    lmax_alm = hp.Alm.getlmax(almE.size)
    if lmax_alm != hp.Alm.getlmax(almB.size):
        raise ValueError('lmax almE ({}) != lmax almB ({})'.format(
            lmax_alm, hp.Alm.getlmax(almB.size)))

    lmax_blm = hp.Alm.getlmax(blmE.size, mmax=blm_mmax)
    if lmax_blm != hp.Alm.getlmax(blmB.size, mmax=blm_mmax):
        raise ValueError('lmax blmE ({}) != lmax blmB ({})'.format(
            lmax_blm, hp.Alm.getlmax(blmB.size, mmax=blm_mmax)))

    almm2, almp2 = eb2spin(almE, almB, inplace=True)
    blmm2, blmp2 = eb2spin(blmE, blmB, inplace=True)
        
    bellp2 = blm2bl(blmp2, m=abs(spin), mmax=blm_mmax, full=True, norm=True, b00=b00)
    bellm2 = blm2bl(blmm2, m=abs(spin), mmax=blm_mmax, full=True, norm=True, b00=b00)

    if lmax_alm >= lmax_blm:
        lmax = lmax_blm
    elif lmax_alm < lmax_blm:
        lmax = lmax_alm
        bellp2 = bellp2[:lmax_alm+1]
        bellm2 = bellm2[:lmax_alm+1]

    if spin >= 0:

        ps_flm = np.zeros((2, almE.size), dtype=almE.dtype)
        ps_flm_p = ps_flm[0] # Plus or E.
        ps_flm_m = ps_flm[1] # Minus or B.

        ps_flm_m[:] = almm2
        ps_flm_p[:] = almp2

        hp.almxfl(ps_flm_m, np.conj(bellm2) * (-1) ** spin, inplace=True)
        hp.almxfl(ps_flm_p, bellm2, inplace=True)

        spin2eb(ps_flm_m, ps_flm_p, spin=spin, inplace=True)        

    if spin <= 0:

        ms_flm = np.zeros((2, almE.size), dtype=almE.dtype)
        ms_flm_p = ms_flm[0] # Plus or E.
        ms_flm_m = ms_flm[1] # Minus or B.

        ms_flm_m[:] = almp2
        ms_flm_p[:] = almm2

        hp.almxfl(ms_flm_m, np.conj(bellp2) * (-1) ** spin, inplace=True)
        hp.almxfl(ms_flm_p, bellp2, inplace=True)

        spin2eb(ms_flm_m, ms_flm_p, spin=spin, inplace=True)        

    if spin == 0:
        # Todo, you can reduce the memory-usage here.
        sflm = np.zeros_like(ps_flm)
        sflm[0] = ps_flm[0]
        sflm[1] = ms_flm[0]
        sflm[1] *= -1
        curvedsky.alm2map(sflm, omap, spin=spin)

    if spin > 0:
        curvedsky.alm2map(ps_flm, omap, spin=spin)        

    if spin < 0:
        curvedsky.alm2map(ms_sflm, omap, spin=spin)        
        omap[1] *= -1

    # Turn coefficients back to E and B.
    spin2eb(almm2, almp2, inplace=True)
    spin2eb(blmm2, blmp2, inplace=True)

    return omap
