'''
Plot the unprocessed I, E, and B beam maps.
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
from mpi4py import MPI

from pixell import enmap, enplot, utils, wcsutils

opj = os.path.join
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_meta(mapdir):
    '''
    Extract freq, pa, season from mapdir string.
    '''

    _, subdir = os.path.split(mapdir)
    for s in subdir.split('_'):
        if s.startswith('f'):
            freq = s
        elif s.startswith('pa'):
            pa = s
        elif s.startswith('s1'):
            season = s
        else:
            continue

    return freq, pa, season

def get_meta_beam(beamfile):
    '''Extract pol, maptype from beamfile.'''

    _, filename = os.path.split(beamfile)
    for s in filename.split('_'):
        if 'weighted_average' in filename:
            maptype = 'weighted_average'
        elif 'model' in filename:
            maptype = 'model'
        pol = filename[-3]

    pol = 'I' if pol == 'T' else pol

    return pol, maptype

def plot_array(arr, outfile):
    '''
    Plot 2d array representing a beam

    Parameters
    ----------
    arr : (3, npix)
        Array containing x, y coords and mapdata.
    outfile : str
        Absolute path to output file.
    '''

    fig, ax = plt.subplots()
    im = ax.imshow(
        np.log10(np.abs(arr[2])), origin='lower', vmin=-4, vmax=0)
    fig.colorbar(im, ax=ax)
    fig.savefig(outfile)
    plt.close()

def arr2wcs(arr):
    '''
    Convert input beam array to CAR WCS.

    Parameters
    ----------
    arr : (3, npix)
        Array containing x, y coords [arcmin] and mapdata.

    Returns
    -------
    wcs : astropy.wcs.wcs.WCS object
        WCS corresponding to array coordinates.
    '''

    # These are in arcmin.
    dec0 = beam[1,0,0]
    dec1 = beam[1,-1,-1]
    ra0 = beam[0,0,0]
    ra1 = beam[0,-1,-1]

    skybox = np.asarray([[ra0, dec0], [ra1, dec1]]) / 60.
    wcs = wcsutils.car(skybox, shape=beam.shape[-2::])

    return wcs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("sdir", help='Beam map directory')
    parser.add_argument("odir", help='Output directory')
    args = parser.parse_args()

    if rank == 0:
        utils.mkdir(args.odir)

    mapdirs = glob.glob(opj(args.sdir, '*'))
    if mapdirs is False:
        if rank == 0:
            raise ValueError('No directories found in {}'.format(args.sdir))

    for mapdir in mapdirs[comm.Get_rank():len(mapdirs)+1:comm.Get_size()]:

        freq, pa, season = get_meta(mapdir)
        outdir = opj(args.odir, '{}_{}_{}'.format(pa, freq, season))
        utils.mkdir(outdir)

        # Find all different temperature map files in subdir.
        files = glob.glob(opj(mapdir, '*', '*T.p'))

        for mapfile in files:

            pol, maptype = get_meta_beam(mapfile)

            beam = np.load(mapfile)
            outname = opj(outdir, 'raw_{}_{}'.format(maptype, pol))
            plot_array(beam, outname)

            wcs = arr2wcs(beam)
            beam_enmap = enmap.zeros((3,) + beam.shape[-2::], wcs, dtype=beam.dtype)
            beam_enmap[0,:] = beam[2]

            # Add E and B modes to enmap
            for cidx, comp in enumerate(['E', 'B']):

                path, filename = os.path.split(mapfile)
                mapfile_pol = glob.glob(opj(path, '*{}*_{}.p'.format(maptype, comp)))[0]

                beam = np.load(mapfile_pol)
                outname = opj(outdir, 'raw_{}_{}'.format(maptype, comp))
                plot_array(beam, outname)

                beam_enmap[1+cidx,:] = beam[2]

            for cidx, comp in enumerate(['I', 'E', 'B']):
                plot = enplot.plot(np.log10(np.abs(beam_enmap[cidx])), ticks=0.1,
                                   colorbar=True)

                outname = opj(outdir, 'enmap_{}_{}'.format(maptype, comp))
                enplot.write(outname, plot)

            outname = opj(outdir, 'beam_{}_ieb.fits'.format(maptype))
            enmap.write_map(outname, beam_enmap)
