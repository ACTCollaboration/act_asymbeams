import numpy as np

from pixell import enmap

def bcast_array(arr, comm, from_rank=0):
    '''
    Wrapper around MPI Bcast.

    Parameters
    ----------
    arr: ndarray, None
        Array to be broadcasted, must exist in scope.
    comm : mpi4py.MPI.Intracomm object
    from_rank : int, optional
        Rank containing array.

    Returns
    -------
    arr : ndarray
        Broadcasted array on each rank.
    '''

    if comm.Get_rank() == from_rank:
        shape = arr.shape
        dtype = arr.dtype
    else:
        shape, dtype = None, None

    shape = comm.bcast(shape, root=from_rank)
    dtype = comm.bcast(dtype, root=from_rank)

    if comm.Get_rank() != from_rank:
        arr = np.zeros(shape, dtype=dtype)

    comm.Bcast(arr, root=from_rank)

    return arr

def bcast_map(imap, comm, root=0):
    '''
    Broadcast enmap.

    Parameters
    ----------
    imap : enmap, None
        Map to be broadcasted, must exist in scope.
    comm : mpi4py.MPI.Intracomm object
    root : int, optional
        Broadcast map from this rank.

    Returns
    -------
    omap : enmap
        Broadcasted map on all ranks.
    '''

    if comm.Get_rank() == root:
        wcs = imap.wcs
    else:
        wcs = None

    wcs = comm.bcast(wcs, root=root)
    omap = bcast_array(np.asarray(imap), comm, from_rank=root)

    return enmap.enmap(omap, wcs)

def reduce_map(imap, comm, root=0):
    '''
    Reduce enmap to root.

    Parameters
    ----------
    imap : enmap
    comm : mpi4py.MPI.Intracomm object
    op : mpi4py.MPI.Op object, optional
        Operation during reduce.
    root : int, optional
        Reduce map to this rank.

    Returns
    -------
    omap : enmap, None
        Reduced map on root, None on other ranks.
    '''

    if comm.Get_rank() == root:
        omap = np.zeros(imap.shape, dtype=imap.dtype)
    else:
        omap = None

    comm.Reduce(np.array(imap), omap, root=root)

    if comm.Get_rank() == root:
        return enmap.enmap(omap, imap.wcs)
    else:
        return None
 
def gather_map(shape, wcs, omap, sel, comm, root=0):
    '''
    Gather distributed slices of map into full map on root rank.

    Parameters
    ----------
    shape : tuple
        Shape of full map.
    wcs : astropy.wcs.wcs.WCS object
        WCS of full map/
    omap : enmap
        slice of full map.
    sel : tuple
        Slice describing map on rank.
    comm : mpi4py.MPI.Intracomm object
    root : int, optional
        Gather onto this rank.

    Returns
    -------
    fmap : enmap, None
        Full map on root rank, None on others
    '''

    size = comm.Get_size()
    shape_diff = len(shape) - len(sel)
    if shape_diff > 0:
        sel = (slice(None, None, None),) * shape_diff + sel

    if comm.Get_rank() == root:
        fmap = enmap.zeros(shape, wcs=wcs, dtype=omap.dtype)
        fmap[sel] = omap
    else:
        fmap = None

    sendranks = list(range(size))
    del sendranks[root]

    comm.Barrier()
    for rank in sendranks:

        if comm.Get_rank() == rank:
            comm.send(sel, dest=root, tag=rank)
            comm.send(omap.shape, dest=root, tag=rank + size)
            comm.Send(np.array(omap), dest=root, tag=rank + 2 * size)

        elif comm.Get_rank() == root:
            osel = comm.recv(source=rank, tag=rank)
            oshape = comm.recv(source=rank, tag=rank + size)
            recvbuff = np.empty(oshape, fmap.dtype)
            comm.Recv(recvbuff, source=rank, tag=rank + 2 * size)
            fmap[osel] = recvbuff

    return fmap

def split_geometry(shape, nslices):
    '''
    Slice a map geometry into bands in theta.

    Parameters
    ----------
    shape : tuple
        Shape of full map.
    nslices : int
        Number of slices.

    Returns
    -------
    slices : list of tuples
        List of slices.
    '''

    pre, shape = shape[:-2], shape[-2:]
    ntheta = shape[0]
    rings = np.arange(ntheta)
    subrings = np.array_split(rings, nslices)

    preslice = slice(None, None, None)
    preslice = (preslice,) * len(pre)
    slices = []

    for subring in subrings:
        start = subring[0]
        end = subring[-1] + 1
        sel = (slice(start, end, None), slice(None, None, None))
        slices.append(preslice + sel)

    return slices

def scatter_map(fmap, comm, root=0, return_slices=False):
    '''
    Scatter slices (in theta) of map.

    Parameters
    ----------
    fmap : enmap, None
        Full map on root, None on other ranks.
    comm : mpi4py.MPI.Intracomm object
    root : int, optional
        Scatter from this rank.
    return_slices : bool
        Also return slices that were used.

    Returns
    -------
    smap : enmap, None
        Sliced map on each rank.
    slices : list of tuples
        List of slices describing maps on ranks. Only
        if return_slices is set
    '''

    size = comm.Get_size()
    if comm.Get_rank() == 0:
        shape = fmap.shape
        wcs = fmap.wcs
        dtype = fmap.dtype
    else:
        shape = None
        wcs = None
        dtype = None

    shape = comm.bcast(shape, root=root)
    wcs = comm.bcast(wcs, root=root)
    dtype = comm.bcast(dtype, root=root)

    slices = split_geometry(shape, comm.Get_size())
    sshape, swcs = enmap.slice_geometry(shape, wcs, slices[comm.Get_rank()])
    smap = enmap.zeros(sshape, wcs=swcs, dtype=dtype)

    recvranks = list(range(size))
    del recvranks[root]

    comm.Barrier()
    for rank in recvranks:

        if comm.Get_rank() == root:
            comm.Send(np.array(fmap[slices[rank]]), dest=rank, tag=rank)

        elif comm.Get_rank() == rank:
            recvbuff = np.array(smap)
            comm.Recv(recvbuff, source=root, tag=rank)
            smap = enmap.enmap(recvbuff, swcs)

    if comm.Get_rank() == root:
        smap = fmap[slices[root]]

    if return_slices:
        return smap, slices
    else:
        return smap

