import numpy as np
import os
import glob

import healpy as hp
from pixell import enmap, curvedsky, utils
from enlib import config, scanutils, scan as enscan
from enact import filedb, actscan, actdata

from act_asymbeams.mpi import bcast_array
from act_asymbeams.spinmaps import trunc_alm
import act_asymbeams.io_errors as errors

def process_input(istr):
    '''
    Return list of (path, meta) pairs that match input.

    Parameters
    ----------
    istr : str
        Path to file, possibly containing {pa}, {freq}, and {season}
        wildcards.

    Returns
    -------
    fileinfos : list
        List of tuples (filepath, (pa, freq, season)) or (filepath, None).

    Raises
    ------
    ValueError
        If no files are found.
        If two or more files share meta information.
    '''

    istr_with_wildcards = istr.replace('{pa}', 'pa*')
    istr_with_wildcards = istr_with_wildcards.replace('{freq}', 'f*')
    istr_with_wildcards = istr_with_wildcards.replace('{season}', 's*')

    paths = glob.glob(istr_with_wildcards)

    metae = []
    filepaths = []
    for path in paths:
        try:
            meta = extract_meta(path, istr)
        except ValueError  as e:
            if len(paths) == 1:
                meta = None
            else:
                raise e

        if meta in metae:
            raise ValueError('Meta info: {} is not unique'.format(meta))

        filepaths.append(path)
        metae.append(meta)

    fileinfos = list(zip(filepaths, metae))
    if len(fileinfos) == 0:
        raise ValueError('No file(s) found for input: {}'.format(istr))

    return fileinfos

def extract_meta(path, template):
    '''
    Return pa, freq and season extracted from input path.

    Parameters
    ----------
    path : str
    template : str
        Same as input path, but may still contain {freq}, {pa} and
        {season} wildcards.

    Returns
    -------
    pa : str
    freq : str
    season : str

    Raises
    ------
    ValueError
        If meta info cannot be extracted.
    '''

    errmsg = 'Meta info cannot be determined for {}'.format(path)
    if path == template:
        raise ValueError(errmsg)

    template = template.replace('{pa}', '{p}')
    template = template.replace('{freq}', '{fr}')
    template = template.replace('{season}', '{s}')

    meta = []
    for wildcard in ['{p}', '{fr}', '{s}']:

        # Get indices into path.
        try:
            start = template.index(wildcard)
        except ValueError:
            raise ValueError(errmsg)
        end = start + len(wildcard)

        meta.append(path[start:end])

    return tuple(meta)

def read_area_geometry(area):
    '''
    Read geometry info corresponding to area footprint.

    Parameters
    ----------
    area : str

    Raises
    ------
    IOError
        If geometry info cannot be read.
    '''

    exts = ['']
    if os.path.splitext(area)[1] == '':
        exts += ['.fits', '.hdf', '.fits.gz']

    for ext in exts:
        try:
            apath = area + ext
            shape, wcs = enmap.read_map_geometry(apath)
            print('Found area file : {}'.format(apath))
            return shape, wcs
        except IOError:
            print('Failed loading area file : {}'.format(apath))

            try:
                apath = os.path.join(config.get('root'), 'area', area + ext)
                shape, wcs = enmap.read_map_geometry(apath)
                print('Found area file : {}'.format(apath))
                return shape, wcs
            except:
                print('Failed loading area file : {}'.format(apath))
                continue

    raise IOError('Cannot find area footprint file for : {}'.format(area))
 
def get_pairs(data, tol):
    '''
    Returns array of paired detectors based on pointing offsets.

    Parameters
    ----------
    data : enlib.DataSet instance
    tol : float
        Pointing tolerance in radians.

    Returns
    -------
    pairs : (npair, 2) array
        Pairs of detector names.
    '''

    tol = tol ** 2
    done = np.zeros(data.ndet, dtype=bool)
    pairs = []
    append = pairs.append

    for i, (x,y) in enumerate(data.point_template):
        for j, (xp, yp) in enumerate(data.point_template):

            if done[j]:
                continue

            if i == j:
                continue

            if (x - xp) ** 2 + (y - yp) ** 2 < tol:
                append([i, j])
                done[i] = True
                break

        done[i] = True

    pairs = data.dets[np.array(pairs)]

    return pairs

def cut_detpairs(data, cut_singles=False):
    '''
    Cut all detectors that are part of an uncut pair.

    Parameters
    ----------
    data : enlib.DataSet instance
    cut_singles : bool, optional
        Instead cut all detectors that have no uncut partner.

    Notes
    -----
    Cut can be done before calibration.
    '''

    pairs = get_pairs(data, 0.2 * utils.arcmin).ravel()

    if cut_singles:
        dets2keep = pairs
    else:
        dets2keep = np.setdiff1d(data.dets, pairs, assume_unique=True)
    
    data.restrict(dets=dets2keep)

def read_scans(ids, idxs, cut_pairs=False, cut_singles=False, downsample=1,
               hwp_resample=False):
    '''
    Try to read scans.

    Parameters
    ----------
    ids : array of strings
        Scan indices (XXXXXXXXXX.XXXXXXXXXX.arX:fXXX).
    idxs : array
        Indices to scan indices array.
    cut_pairs : bool, optional
        Cut detectors that are part of an uncut pair.
    cut_singles : bool, optional
        Cut detectors that have no uncut partner.
    downsample : int, optional
        Downsample data by this factor.
    hwp_resample : bool
        Resample data is HWP is found.

    Returns
    -------
    idxs_read : list
        Indices to succesfullly read scans.
    scans_read : list
        Succesfully read scans.

    Notes
    -----
    Similar to enlib.scanutils.read_scans but allows for pair cuts.
    '''

    idxs_read = []
    scans_read = []

    for idx in idxs:
        try:
            entry = filedb.data[ids[idx]]
            data = actdata.read(entry, verbose=False, exclude=['tod'])
            data = actdata.calibrate(data, verbose=False)

            if cut_pairs:
                cut_detpairs(data)
            if cut_singles:
                cut_detpairs(data, cut_singles=True)
            
            scan = actscan.ACTScan(entry, d=data, verbose=False)

        except errors.DataMissing as e:
            continue
            
        hwp_active = np.any(scan.hwp_phase[0] != 0)
        if hwp_resample and hwp_active:
            mapping = enscan.build_hwp_sample_mapping(scan.hwp)
            scan = scan.resample(mapping)
        scan = scan[:,::downsample]

        idxs_read.append(idx)
        scans_read.append(scan)

    return idxs_read, scans_read

def distribute_scans(ids, comm, cut_pairs=False, cut_singles=False):
    '''
    Give each rank subset of scans such that ntod * ndet is distributed evenly.

    Parameters
    ----------
    ids : array of strings
        Scan indices (XXXXXXXXXX.XXXXXXXXXX.arX:fXXX).
    comm : mpi4py.MPI.Intracomm object
    cut_pairs : bool, optional
        Cut detector pairs.
    cut_singles : bool, optional
        Cut detectors without a partner.

    Returns
    -------
    scans : array of strings
        Scans for this rank.

    Raises
    ------
    AllCutError
        If no scans remain after autocuts.
    '''

    idxs_loc = np.arange(ids.size)[comm.Get_rank()::comm.Get_size()]

    idxs_loc, scans_loc = read_scans(ids, idxs_loc, downsample=config.get("downsample"),
                                     cut_pairs=cut_pairs, cut_singles=cut_singles)
    idxs_loc = np.asarray(idxs_loc, dtype=int)

    # Prune fully autocut scans.
    dets_loc  = [len(scan.dets) for scan in scans_loc]
    idxs_loc  = [idx  for idx, ndet in zip(idxs_loc, dets_loc) if ndet > 0]
    scans_loc = [scan for scan, ndet in zip(scans_loc, dets_loc) if ndet > 0]

    if idxs_loc == []:
        idxs_loc = [-1] # Beacuse we cannot allgatherv empty lists.
    read_ids = []
    for idx in utils.allgatherv(idxs_loc, comm):
        if idx == -1:
            continue
        read_ids.append(ids[idx])

    read_ntot = len(read_ids)
    if read_ntot == 0:
        raise errors.AllCutError('No scans remain after autocuts.')

    costs_loc = [s.nsamp * s.ndet for s in scans_loc]
    idxs_loc = scanutils.distribute_scans(idxs_loc, costs_loc, None, comm)
    del scans_loc

    # Reread the correct files this time.
    _, scans_loc = read_scans(ids, idxs_loc, downsample=config.get("downsample"),
                              cut_pairs=cut_pairs, cut_singles=cut_singles)

    return scans_loc

def read_ps(ps_file, lmax):
    '''
    Read .npy file and return power spectrum.

    Parameters
    ----------
    ps_file : str
        Absolute path to power spectrum file.
    lmax : int
        lmax of output.

    Returns
    -------
    ps : array
        TT power spectrum from 0 to lmax.

    Notes
    -----
    We assume .npy file has shape (lmax+1,) or (N,lmax+1). Only first row is
    used in latter case.
    '''

    cl = np.load(ps_file)
    try:
        cl = cl[0,:]
    except IndexError:
        pass
    lmax_in = cl.size - 1
    cl_out = np.zeros(lmax+1, dtype=cl.dtype)

    if lmax >= lmax_in:
        cl_out[:lmax_in+1] = cl
    else:
        cl_out[:] = cl[:lmax+1]

    return cl_out

def get_alm(lmax, comm, ps_file=None, alm_file=None, seed=None, root=0):
    '''
    Return alm array on all ranks.

    Parameters
    ----------
    lmax : int
        Maximum multipole.
    comm : mpi4py.MPI.Intracomm object
    ps_file : str, optional
        Path to power spectrum .npy file.
    alm_file : str, optional
        Path to alm .fits file.
    seed : int, optional
        Seed to numpy random number generator.
    root : int, optional
        MPI root rank.

    Returns
    -------
    alm : (npol, nelem) array
        Alm array, same on all ranks.

    Raises
    ------
    ValueError
        If no power spectrum or alm file is given.
    '''

    if bool(ps_file) is bool(alm_file):
        raise ValueError('Either power spectrum or alm file required.')

    if ps_file is not None:
        if comm.Get_rank() == root:
            ps = read_ps(ps_file, lmax)
        else:
            ps = None
        ps = bcast_array(ps, comm, from_rank=root)
        alm = curvedsky.rand_alm(ps, seed=seed)

    elif alm_file is not None:
        if comm.Get_rank() == root:
            try:
                alm, mmax = hp.fitsfunc.read_alm(alm_file, hdu=(1, 2, 3), 
                                                 return_mmax=True)
            except IndexError:
                # Assume T-only.
                alm, mmax = hp.fitsfunc.read_alm(alm_file, hdu=1, 
                                                 return_mmax=True)
                alm = alm[np.newaxis,:]

            alm = alm.astype(np.complex128)
            lmax_alm = hp.Alm.getlmax(alm.shape[1], mmax=mmax)
            if lmax_alm > lmax:
                alm = trunc_alm(alm, lmax)
        else:
            alm = None
        alm = bcast_array(alm, comm, from_rank=root)

    return alm

def get_outdir(basepath, meta=None):
    '''
    Return path to output directory.

    Parameters
    ----------
    basepath : str
        Path to parent directory.
    meta : tuple, optional
        Strings of meta info e.g. ('pa1', 'f150', 's13').    

    Returns
    -------
    outdir : str
        Path to output directory.
    '''
    try:
        outdir = os.path.join(basepath, '{}_{}_{}'.format(*meta))
    except TypeError as e:
        if meta is None: 
            outdir = basepath
        else: 
            raise e

    return outdir

def files_exist(files):
    '''
    Return True if all requested files exist.

    Parameters
    ----------
    files : array-like
        Paths to files.
    
    Returns
    -------
    exist : bool
    '''

    for f in files:
        if not os.path.isfile(f):
            return False

    return True

def remove_existing(fileinfos, onames, basepath):
    '''
    Remove entries that refer to existing files.

    Parameters
    ----------
    fileinfos : list
        List of tuples (filepath, (pa, freq, season)) or (filepath, None).
    onames : list
        List of filenames to check for.
    basepath : str
        Directory to look in.

    Returns
    -------
    fileinfos_keep : list
        Copy of input list with only nonexisting entries.
    fileinfos_cut : list
        Copy of input list with only existing entries.
    '''

    fileinfos_cut = []
    fileinfos_keep = []

    for fileinfo, meta in fileinfos:

        outdir = get_outdir(basepath, meta=meta)
        files = [os.path.join(outdir, oname) for oname in onames]

        if not files_exist(files):
            fileinfos_keep.append((fileinfo, meta))
        else:
            fileinfos_cut.append((fileinfo, meta))

    return fileinfos_keep, fileinfos_cut
