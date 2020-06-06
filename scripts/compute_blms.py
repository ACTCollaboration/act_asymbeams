import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import traceback
import glob
import argparse
from mpi4py import MPI

import healpy as hp
from pixell import enmap, curvedsky, utils, enplot, coordinates, wcsutils
from act_asymbeam import io

opj = os.path.join
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def butter_filter(freqs, freqc, order=7):
    '''
    Return a Butterworth filter.

    gain = 1 / (1 + (freqs / freqc)^(2 * order)

    Parameters
    ----------
    freqs : (nfreq) array
    freqc : float
        Critial frequency at which the gain is 0.5.
    order : int, optional
        Order of filter. Higher order means more rectangular filter.

    Returns
    -------
    gain : (nfreq) array
        The gain of the filter.
    '''

    gain = freqs ** (2 * order)
    gain /= float(freqc ** (2 * order))
    gain += 1.
    gain **= -1

    return gain

def smooth_butter(imap, lmax, order=7):
    '''
    Flat-sky smoothing of map using Butterworth filter.

    Parameters
    ----------
    bmap : enmap
    lmax : scalar
        Critial frequency of filter (i.e. where gain = 0.5).
    order : int, optional
        Order or filter.
    '''

    fmap = enmap.fft(imap)
    lmap = enmap.modlmap(imap.shape, imap.wcs)
    gain = butter_filter(lmap, lmax, order=order)
    fmap *= gain
    return enmap.ifft(fmap).real

def allocate_car_strip(lmax, radius):
    '''
    Create empty (north) polar cap map sutable for SHT operations.

    Parameters
    ----------
    lmax : int
        Determines the resolution of output map.
    radius : float
        Radius of cap in radians.
    '''

    ntheta = int(lmax + 1)
    res = [np.pi / float(ntheta - 1), 2 * np.pi / float(ntheta)]
    pos = [[np.pi / 2 - radius, -np.pi], [np.pi / 2, np.pi]]
    oshape, owcs = enmap.geometry(pos=pos, res=res, proj="car")
    return enmap.zeros(oshape, owcs)

def allocate_zea_cap(radius, res):
    '''
    Allocate north polar cap in ZEA projection.

    Parameters
    ----------
    radius : float
        Radius of polar cap.
    res : (2,) array
        Dec and RA esolution in rad / pix.

    Returns
    -------
    omap : enmap
    '''

    wo  = wcsutils.WCS(naxis=2)
    wo.wcs.ctype = ["RA---ZEA","DEC--ZEA"]
    wo.wcs.crval = [0, 90]
    wo.wcs.cdelt = np.degrees(res)[::-1]
    wo.wcs.crpix = [1, 1]
    x, y = wo.wcs_world2pix(0, np.degrees(np.pi / 2. - radius), 1)
    y = int(np.ceil(y))
    n = 2 * y - 1
    wo.wcs.crpix = [y, y]

    return enmap.zeros((n, n), wo)

def apodized_disk(imap, rmap, radius, width):
    '''
    Cut out disk and apodize with cosine apodization.

    Parameters
    ----------
    imap : enmap
    rmap : enmap
    radius : float
    width : float
    '''

    imap[rmap > radius] = 0
    inner_r = radius - width
    mask = rmap > inner_r
    width = float(width)
    imap[mask] *= (1 + np.cos(-np.pi * inner_r / width + np.pi * rmap[mask] / width))
    imap[mask] *= 0.5

def rotate_to_pole(imap, omap):
    '''
    Rotate input map into output map.

    Pararameters
    ------------
    imap : enmap
        Input map.
    omap : enmap
        Output map.
    '''

    opos = enmap.posmap(omap.shape, omap.wcs)
    coords = np.radians([[0, 0]])
    from_sys = 'cel'
    to_sys = ["cel",[[0,np.pi/2,coords[0,1],coords[0,0]],False]]
    iposs = coordinates.transform(from_sys, to_sys, opos[::-1], pol=None)
    ipos, rest = iposs[::-1], iposs[2:]
    omap[:] = imap.at(ipos, order=3)

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convert beam map(s) to spherical harmonic coefficients (blms).')
    parser.add_argument("ipath",
        help='Path to beam .fits file, can contain {pa}, {freq}, {season} wildcards')
    parser.add_argument("odir",
        help='Output directory where blm.fits file is stored. paXX_fXXX_sXX subdirs '
             'are created depending on input.')
    parser.add_argument("--lmax", type=int, default=25000,
        help='Maximum multipole.')
    parser.add_argument("--mmax", type=int, default=100,
        help='Maximum m-mode writted to disk')
    args = parser.parse_args()

    if mpi_rank == 0:
        try:
            fileinfos = io.process_input(args.ipath)
        except ValueError as e:
            traceback.print_exc()
            fileinfos = None
    else:
        fileinfos = None

    fileinfos = mpi_comm.bcast(fileinfos, root=0)
    if fileinfos is None:
        raise ValueError()

    if mpi_rank == 0:
        print('Found {} file(s):'.format(len(fileinfos)))
        for fi in fileinfos:
            print('  {}'.format(fi[0]))

    if mpi_rank == 0:
        utils.mkdir(args.odir)

    for filepath, meta in fileinfos[mpi_rank:len(fileinfos)+1:mpi_size]:
        try:
            pa, freq, season = meta
            outdir = opj(args.odir, '{}_{}_{}'.format(*meta))
            utils.mkdir(outdir)
        except TypeError:
            outdir = args.odir

        # We are only dealing with I beams for now.
        try:
            beam = enmap.read_map(filepath, sel=np.s_[0,:,:])
        except IndexError:
            beam = enmap.read_map(filepath, sel=np.s_[:,:])

        outname = opj(outdir, 'raw_I')
        plot = enplot.plot(np.log10(np.abs(beam)),
                           ticks=.1, colorbar=True, quantile=0, min=-4)
        enplot.write(outname, plot)

        rmap = enmap.modrmap(beam.shape, beam.wcs)
        radius = rmap[0,beam.shape[1] // 2]

        omap = allocate_car_strip(args.lmax, radius)

        # Smooth if projection results in decimation of beam.
        if omap.wcs.wcs.cdelt[1] > min(beam.wcs.wcs.cdelt):
            beam = smooth_butter(beam, args.lmax)

        apodized_disk(beam, rmap, radius, radius / 5.)

        outname = opj(outdir, 'smoothed_I')
        plot = enplot.plot(np.log10(np.abs(beam)),
                           ticks=.1, colorbar=True, quantile=0, min=-4)
        enplot.write(outname, plot)

        rotate_to_pole(beam, omap)

        outname = opj(outdir, 'proj_I')
        plot = enplot.plot(np.log10(np.abs(omap)),
                           ticks=[5, 20], colorbar=True, quantile=0, min=-4)
        enplot.write(outname, plot)

        blm = curvedsky.map2alm_cyl(omap, lmax=args.lmax)
        omap2 = curvedsky.alm2map(blm, enmap.zeros(omap.shape, omap.wcs))

        outname = opj(outdir, 'out_I')
        plot = enplot.plot(np.log10(np.abs(omap - omap2)),
                           ticks=[5, 20], colorbar=True, quantile=0, min=-4)
        enplot.write(outname, plot)

        res = np.radians(beam.wcs.wcs.cdelt)[::-1]
        omap_zea = allocate_zea_cap(radius, res)
        omap_zea = curvedsky.alm2map(blm, omap_zea)

        outname = opj(outdir, 'out_zea_I')
        plot = enplot.plot(np.log10(np.abs(omap_zea)),
                           colorbar=True, ticks=[5,20], quantile=0, min=-4)
        enplot.write(outname, plot)

        outname = opj(outdir, 'blm.fits')
        try:
            hp.fitsfunc.write_alm(outname, blm, mmax=args.mmax)
        except IOError:
            os.remove(outname)
            hp.fitsfunc.write_alm(outname, blm, mmax=args.mmax)
