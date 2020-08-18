import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import numpy as np
import os
import glob
import argparse
from mpi4py import MPI

import healpy as hp
from pixell import enmap, curvedsky, enplot, utils, sharp
from act_asymbeams.io import process_input

from compute_blms import allocate_zea_cap

opj = os.path.join
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Make summary plots of input blms.')
    parser.add_argument("ipath",
        help='Path to blm .fits file, can contain {pa}, {freq}, {season} wildcards')
    parser.add_argument("odir",
        help='Output directory where figures are stored. paXX_fXXX_sXX subdirs '
             'are created depending on input.')
    args = parser.parse_args()

    if mpi_rank == 0:
        try:
            fileinfos = process_input(args.ipath)
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

        blm, mmax = hp.fitsfunc.read_alm(filepath, return_mmax=True)
        lmax = hp.Alm.getlmax(blm.size, mmax=mmax)
        
        ntheta = int(8 * lmax + 1)
        res = [np.pi / float(ntheta - 1)] * 2
        radius = np.radians(0.2)
        omap_zea = allocate_zea_cap(radius, res)        
        ainfo = sharp.alm_info(lmax, mmax=mmax, nalm=blm.size)

        enplot_opts = {'nolabels' : True,
                       'layers' : True,
                       'ticks' : [0.1, 45],
                       'nstep' : 200,
                       'no_image' : True}

        dpi = 350
        figsize = (omap_zea.shape[1] * 2.2 / float(dpi), omap_zea.shape[0] * 5 / float(dpi))
        cmap = plt.get_cmap('RdBu_r')
        cmap_half = truncate_colormap(cmap, 0.5, 1)

        fig2 = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        spec2 = gridspec.GridSpec(ncols=8, nrows=18, figure=fig2)
        axr = []
        axr.append(fig2.add_subplot(spec2[:4,:4]))
        axr.append(fig2.add_subplot(spec2[:4,4:]))
        cax0 = fig2.add_subplot(spec2[4,:])
        axr.append(fig2.add_subplot(spec2[5:9,:4]))
        axr.append(fig2.add_subplot(spec2[5:9,4:]))
        axr.append(fig2.add_subplot(spec2[9:13,:4]))
        axr.append(fig2.add_subplot(spec2[9:13,4:]))
        axr.append(fig2.add_subplot(spec2[13:17,:4]))
        axr.append(fig2.add_subplot(spec2[13:17,4:]))
        cax1 = fig2.add_subplot(spec2[17,:])        
        
        omap_zea = curvedsky.alm2map(blm, omap_zea, ainfo=ainfo)
        plot = enplot.plot(np.log10(np.abs(omap_zea)), **enplot_opts)

        im = axr[0].imshow(10 * np.log10(np.abs(omap_zea)), cmap=cmap_half, vmin=-40, vmax=0)
        axr[0].imshow(np.asarray(plot[0].img))

        for midx, m in enumerate(range(0, min(7, mmax + 1))):

            blmc = blm.copy()
            start = hp.Alm.getidx(lmax, m, m)
            end = start + lmax + 1 - m
            blmc[:start] *= 0
            blmc[end:] *= 0
            
            omap_zea = curvedsky.alm2map(blmc, omap_zea * 0, ainfo=ainfo)
            plot = enplot.plot(omap_zea, **enplot_opts)

            if m == 0:
                axr[midx+1].imshow(10 * np.log10(np.abs(omap_zea)), cmap=cmap_half, vmin=-40, vmax=0)
            else:
                # In dB units.
                vmax = -20
                vmin = -45                
                posm = omap_zea.copy()
                posm[posm < 10 ** (vmin / 10.)] = 10 ** -20
                posm = 10 * np.log10(posm)

                negm = omap_zea.copy()
                negm[negm > 0] = 0
                negm *= -1                
                negm[negm < 10 ** (vmin / 10.)] = 10 ** (vmin / 10.)
                negm = 10 * np.log10(negm)
                negm = -negm + 2 * vmin
                
                posm[posm == -200] = negm[posm == -200]
                totm = posm
                
                if m == 1:                
                    im2 = axr[midx+1].imshow(totm, cmap=cmap, vmin=-70, vmax=-20)
                else:
                    axr[midx+1].imshow(totm, cmap=cmap, vmin=-70, vmax=-20)

            axr[midx+1].imshow(np.asarray(plot[0].img))            
            bbox_props = dict(boxstyle="square, pad=0.3", lw=0.5, fc="w", ec="0.5", alpha=0.8)
            axr[midx+1].text(0.970, 0.970, r'$m={}$'.format(m), bbox=bbox_props, fontsize=10,
                             transform=axr[midx+1].transAxes, va='top', ha='right')


        for ax in axr:
            ax.tick_params(top=False, bottom=False, left=False, right=False)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        
        cbar0 = fig2.colorbar(im, cax=cax0, orientation='horizontal', extend='min', 
                              ticks=[-40, -30, -20, -10, 0])
        cbar1 = fig2.colorbar(im2, cax=cax1, orientation='horizontal', extend='both',
                              ticks=[-70, -60, -50, -40, -30, -20])
        cbar1.ax.set_xticklabels(['(-20)', '(-30)', '(-40)', '-40', '-30', '-20'])
        
        cax0.set_xlabel(r'$\mathrm{dBi}$')
        cax1.set_xlabel(r'$\mathrm{dBi}$')
        fig2.suptitle('{}_{}_{}'.format(*meta))

        outname = opj(outdir, 'out_zea_I_m_{}_{}_{}_{}'.format(mmax, *meta))        
        fig2.savefig(outname)

        # Plot bells.
        ells = np.arange(lmax + 1)
        q_ell = np.sqrt(4 * np.pi / (2 * ells + 1))

        blm = hp.almxfl(blm, q_ell, mmax=mmax, inplace=True)
        blm /= blm[0]

        outname = opj(outdir, 'bell_m_{}_{}_{}'.format(*meta))        
        fig, axs = plt.subplots(nrows=2, figsize=(6, 7.5), sharex=True)
        # Plot m = 0 seperately.
        axs[0].plot(ells, np.real(blm[:lmax+1]), label=r'$m=0$')
        axs[0].set_ylabel(r'$\sqrt{4\pi / (2 \ell + 1)} b^I_{\ell 0}$')
        axs[0].set_title('{}_{}_{}'.format(*meta))
        axs[0].legend(frameon=False)

        # Plot m > 0.
        for m in range(1, min(6, mmax + 1)):
            start = hp.Alm.getidx(lmax, m, m)
            end = start + lmax + 1 - m
            bell_m = np.abs(blm[start:end])
            axs[1].plot(ells[m:], bell_m[:lmax+1], label=r'$m={}$'.format(m),
                        color='C{}'.format(m))            
        axs[1].legend(frameon=False)
        axs[1].set_xlabel(r'Multipole $\ell$')
        axs[1].set_ylabel(r'$\sqrt{4\pi / (2 \ell + 1)} |b^I_{\ell m}|$')
        for ax in np.ravel(axs):
            ax.tick_params(direction='in', right=True, top=True)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        fig.savefig(outname, dpi=250)
        plt.close(fig)
        
        # Save text file with ell and m=0 beam.
        header = ['ell', 'B_ell']
        fmts = '\t'.join(['%18s'] * len(header))  
        outname = opj(outdir, 'beam_{}_{}_{}.txt'.format(*meta))        
        np.savetxt(outname, np.stack([ells.astype(float), np.real(blm[:lmax+1])]).T,
                   header=fmts%tuple(header), fmt='%12.12e', delimiter='\t')
        
        
