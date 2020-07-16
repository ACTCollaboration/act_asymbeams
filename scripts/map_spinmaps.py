#!/usr/bin/env python
import numpy as np
import os
import traceback

import healpy as hp
from pixell import enmap, enplot, utils, fft
from enlib import config, pmat, array_ops
from enact import filedb, nmat_measure

from act_asymbeams import spinmaps
from act_asymbeams import io
from act_asymbeams import mpi
from act_asymbeams import io_errors as errors

from mpi4py import MPI

opj = os.path.join
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if __name__ == '__main__':

    parser = config.ArgumentParser(os.environ["HOME"] + "./enkirc",
        description='Convolve input alm with asymmetric beam and map on sky.')
    parser.add_argument("ipath_blm",
        help='Path to blm .fits file, can contain {pa}, {freq}, {season} wildcards')
    parser.add_argument("odir",
        help='Output directory where maps and figures are stored. paXX_fXXX_sXX subdirs '
        'are created depending on input.')
    parser.add_argument("area", type=lambda s: [item for item in s.split(',')],
        help='Comma-seperated list of area names or paths.')
    parser.add_argument("--smap-alm", default=None,
        help='Path to alm .fits file')
    parser.add_argument("--smap-ps", default=None,
        help='Path to power spectrum .npy file')
    parser.add_argument("--smap-seed", default=0, type=int,
        help='Seed for alm generation if power spectrum is given.')
    parser.add_argument("--smap-lmax", default=15000, type=int,
        help='Maximum multipole used for beam convolution.')
    parser.add_argument("--smap-sel", default='',
        help='Selection string to enact.filedb.')
    parser.add_argument("--smap-write-div", action="store_true",
        help='Write div map to disk')
    parser.add_argument("--smap-write-rhs", action="store_true",
        help='Write rhs map to disk')
    parser.add_argument("--smap-tag", default=None,
        help='Tag appended to ouput files.')
    parser.add_argument("--smap-asymmetric-only", action="store_true",
        help='Scan using only azimuthally asymmetric part of the beam.')
    parser.add_argument("--smap-symmetric-only", action="store_true",
        help='Scan using only azimuthally symmetric part of the beam.')
    parser.add_argument("--smap-pairs-only", action="store_true",
        help='Scan and map using only paired detectors.')
    parser.add_argument("--smap-singles-only", action="store_true",
        help='Scan and map using only detectors without partner.')
    parser.add_argument("--no-noise-weight", action="store_true",
        help='Do not weight detectors by their white noise level.')
    parser.add_argument("--smap-continue", action="store_true",
        help='Do not recompute maps if output already exists.')

    args = parser.parse_args()

    dtype = np.float32 if config.get("map_bits") == 32 else np.float64

    if bool(args.smap_alm) == bool(args.smap_ps):
        raise ValueError('Either "--alm" or "--ps" argument is required.')

    if mpi_rank == 0:
        try:
            fileinfos = io.process_input(args.ipath_blm)            
        except: 
            traceback.print_exc()
            fileinfos = None
    else:
        fileinfos = None

    fileinfos = mpi_comm.bcast(fileinfos, root=0)
    if fileinfos is None:
        raise ValueError()

    if mpi_rank == 0:
        print('Creating maps for:')
        print('  beam(s):')
        for fi in fileinfos:
            print('    {}'.format(fi[0]))
        print('  area(s)')
        for area in args.area:
            print('    {}'.format(area))

    if mpi_rank == 0:
        utils.mkdir(args.odir)

    area_tags = [os.path.splitext(os.path.split(a)[-1])[0] for a in args.area]
    out_tag = '_{}'.format(args.smap_tag) if args.smap_tag is not None else ''
    
    outnames = {'spinmaps' : lambda tag: 'spinmaps_{}.fits'.format(tag),
                'omap' : lambda tag: 'omap_{}.fits'.format(tag)}
    if args.smap_write_rhs:
        outnames['rhs'] = lambda tag: 'rhs_{}.fits'.format(tag)
    if args.smap_write_div:
        outnames['div'] = lambda tag: 'div_{}.fits'.format(tag)

    if mpi_rank == 0:
        fileinfos_per_area = {}

        for area_tag in area_tags:
            if args.smap_continue:
                tag = area_tag + out_tag
                onames = [outnames[otype](tag) for otype in outnames.keys()]
                fileinfos_keep, fileinfos_cut = io.remove_existing(
                    fileinfos, onames, args.odir)
                for _, meta in fileinfos_cut: 
                    print('Skipping : {}, reason : output already exists.'.
                          format(tag if meta is None else meta + (tag,)))
            else:
                fileinfos_keep = fileinfos
            fileinfos_per_area[area_tag] = fileinfos_keep
    else:
        fileinfos_per_area = None
    fileinfos_per_area = mpi_comm.bcast(fileinfos_per_area, root=0)

    for area, area_tag in zip(args.area, area_tags):

        if not fileinfos_per_area[area_tag]:
            continue

        if mpi_rank == 0:
            shape, wcs = io.read_area_geometry(area)
        else:
            shape, wcs = None, None
        shape = mpi_comm.bcast(shape, root=0)
        wcs = mpi_comm.bcast(wcs, root=0)

        slices = mpi.split_geometry(shape, mpi_size)
        sshape, swcs = enmap.slice_geometry(shape, wcs, slices[mpi_rank])

        for filepath, meta in fileinfos_per_area[area_tag]:

            outdir = io.get_outdir(args.odir, meta=meta)
            if mpi_rank == 0:
                utils.mkdir(outdir)
                             
            filedb.init()
            sel = '{},'.format(area_tag) + args.smap_sel
            if meta is not None:
                sel = '{},{},{},'.format(*meta) + sel
            ids = filedb.scans[sel]
            if ids.size == 0:
                if mpi_rank == 0:
                    print('Skipping : {}, reason : No scans found'.format(sel))
                continue

            if mpi_rank == 0:
                blm, mmax = hp.fitsfunc.read_alm(filepath, return_mmax=True)
                blm = blm.astype(np.complex128)
            else:
                blm, mmax = None, None
            mmax = mpi_comm.bcast(mmax, root=0)
            blm = mpi.bcast_array(blm, mpi_comm)

            if args.smap_lmax > hp.Alm.getlmax(blm.size, mmax=mmax):
                raise ValueError('Requested lmax: {} exceeds beam lmax: {}'
                    .format(args.smap_lmax, hp.Alm.getlmax(blm.size, mmax=mmax)))

            alm = io.get_alm(args.smap_lmax, mpi_comm, seed=args.smap_seed,
                             ps_file=args.smap_ps, alm_file=args.smap_alm)

            smap = enmap.zeros((3,) + sshape[-2::], wcs=swcs, dtype=dtype)
            if not args.smap_asymmetric_only:
                spinmaps.compute_spinmap_real(alm, blm, mmax, 0, smap[0])
            if not args.smap_symmetric_only:
                spinmaps.compute_spinmap_real(alm, blm, mmax, 2, smap[1:])

            del alm
            del blm

            spinmaps = mpi.gather_map((3,) + shape[-2::], wcs, smap,
                            slices[mpi_rank], mpi_comm, root=0)
            del smap

            if mpi_rank == 0:
                outname = opj(outdir, outnames['spinmaps'](area_tag + out_tag))
                enmap.write_map(outname, spinmaps)

            spinmaps = mpi.bcast_map(spinmaps, mpi_comm, root=0)

            if mpi_rank == 0:
                print('Reading scans for : {}'.format(sel))

            try:
                scans_local = io.distribute_scans(ids, mpi_comm,
                                                  cut_pairs=args.smap_singles_only,
                                                  cut_singles=args.smap_pairs_only)

            except errors.AllCutError as e:
                if mpi_rank == 0:
                    print('Skipping : {}, reason : '.format(sel) + str(e))
                continue

            # omap, rhs, div = scan_and_bin(scans_local, spinmaps, args.no_noise_weight, mpi_comm)
            # omap, rhs, div = scan_and_map(scans_local, spinmaps, cg_steps, mpi_comm)

            rhs = enmap.zeros((3,) + shape[-2::], wcs=wcs, dtype=dtype)
            tmp = enmap.zeros((3,) + shape[-2::], wcs=wcs, dtype=dtype)
            div = enmap.zeros((3, 3) + shape[-2::], wcs=wcs, dtype=dtype)
            for sidx, scan in enumerate(scans_local):
                print('[rank {:4d}]: mapping {}/{} : {}'.format(
                    mpi_rank, sidx + 1, len(scans_local), scan))

                if args.no_noise_weight:
                    tod = np.zeros((scan.ndet, scan.nsamp), dtype=dtype)
                else:
                    tod = scan.get_samples()
                    tod = tod.astype(dtype)
                    ft = fft.rfft(tod)
                    ft *= np.sqrt(scan.nsamp)
                    noise = nmat_measure.detvecs_jon(ft, scan.srate)
                    del ft

                pmap = pmat.PmatMap(scan, rhs[0])
                pcut = pmat.PmatCut(scan)
                junk = np.zeros(pcut.njunk, dtype=dtype)

                # Save polangle, set polangle to zero for map2tod.
                det_comps = pmap.scan.comps.copy()
                hack_comps = np.zeros_like(det_comps)
                hack_comps[:,:2] = 1.
                pmap.scan.comps = hack_comps

                pmap.forward(tod, spinmaps, tmul=0.)

                # Restore polangle for map2tod.
                pmap.scan.comps = det_comps

                if not args.no_noise_weight:
                    noise.white(tod)
                pcut.backward(tod, junk)
                pmap.backward(tod, rhs)

                for i in range(3):
                    tmp *= 0
                    tmp[i] = 1
                    pmap.forward(tod, tmp, tmul=0.) # Overwrite TOD.
                    if not args.no_noise_weight:
                        noise.white(tod)
                    pcut.backward(tod, junk)
                    pmap.backward(tod, div[i])

                if not args.no_noise_weight:
                    del noise

            del tmp
            del scans_local
            del spinmaps

            mpi_comm.Barrier()
            if mpi_comm.Get_rank() == 0:
                print('Reducing...')

            rhs = mpi.reduce_map(rhs, mpi_comm, root=0)
            div = mpi.reduce_map(div, mpi_comm, root=0)

            if mpi_comm.Get_rank() == 0:
                print('Scattering...')

            srhs, slices = mpi.scatter_map(rhs, mpi_comm, root=0, return_slices=True)
            sdiv = mpi.scatter_map(div, mpi_comm, root=0)

            if mpi_comm.Get_rank() == 0:
                print('Solving...')

            isdiv = array_ops.eigpow(sdiv, -1, axes=[0,1], copy=False,
                                     lim=1e-5, fallback="scalar")
            smap  = enmap.map_mul(isdiv, srhs)
            fmap = mpi.gather_map((3,) + shape[-2::], wcs, smap, slices[mpi_rank],
                              mpi_comm, root=0)

            if mpi_comm.Get_rank() == 0:
                print('Saving...')

            if mpi_rank == 0:
                outname = opj(outdir, outnames['omap'](area_tag + out_tag))
                enmap.write_map(outname, fmap)

                if args.smap_write_rhs:
                    outname = opj(outdir, outnames['rhs'](area_tag + out_tag))
                    enmap.write_map(outname, rhs)

                if args.smap_write_div:
                    outname = opj(outdir, outnames['div'](area_tag + out_tag))
                    enmap.write_map(outname, div)

            if mpi_comm.Get_rank() == 0:
                print('Done with : {}.'.format(sel))
