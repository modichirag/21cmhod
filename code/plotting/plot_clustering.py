#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
#

from matplotlib import rc, rcParams, font_manager
rcParams['font.family'] = 'serif'
fsize = 12
fontmanage = font_manager.FontProperties(family='serif', style='normal',
    size=fsize, weight='normal', stretch='normal')
font = {'family': fontmanage.get_family()[0],
        'style':  fontmanage.get_style(),
        'weight': fontmanage.get_weight(),
        'size': fontmanage.get_size(),
        }

print(font)


#
import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-s', '--size', help='which box size simulation', default='small')
args = parser.parse_args()


model = 'ModelA' #args.model
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
dpathxi = '../../data/outputs/%s/%s/'%(suff, model)
if boxsize == 'big':
    suff = suff + '-big'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass




def make_bias_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [2.0,6.0,4.0]

    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(7,3.5))

    for iz, zz in enumerate(zlist):

        aa = 1.0/(1.0+zz)
        # Put on a fake symbol for the legend.

        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))
        bka = pkd[:,2]
        bkx = pkd[:,1]
        ax[0].plot(pkd[:,0],bka,'C%d-'%iz,label="z={:.1f}".format(zz))
        ax[0].plot(pkd[:,0],bkx,'C%d--'%iz,)

        #ax[1].plot([100],[100],'s',color=col,label="z={:.1f}".format(zz))
        # Plot the data, xi_mm, xi_hm, xi_hh

        xim = np.loadtxt(dpathxi + "ximatter_{:06.4f}.txt".format(aa))
        xix = np.loadtxt(dpathxi + "ximxh1mass_{:06.4f}.txt".format(aa))
        xih = np.loadtxt(dpathxi + "xih1mass_{:06.4f}.txt".format(aa))

        # and the inferred biases.
        ba = np.sqrt(ius(xih[:,0],xih[:,1])(xim[:,0])/xim[:,1])
        bx = ius(xix[:,0], xix[:,1])(xim[:,0])/xim[:,1] 
        #ba = np.sqrt(xih[:,1]/xim[:,1])
        #bx = xix[:,1]/xim[:,1]
        xx = np.array([i.mean() for i in np.array_split(xim[:,0], np.arange(2, 28, 2))])
        ba = [i.mean() for i in np.array_split(ba, np.arange(2, 28, 2))]
        bx = [i.mean() for i in np.array_split(bx, np.arange(2, 28, 2))]
        
        if iz==0:
            ax[1].plot(xx*-1,ba,'k-', label=r'$b_a$')
            ax[1].plot(xx*-1,bx,'k--', label=r'$b_x$')
        ax[1].plot(xx,ba,'C%d-'%iz)
        ax[1].plot(xx,bx,'C%d--'%iz)

        # put on a line for Sigma -- labels make this too crowded.
        pk  = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        sig = np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        ax[0].axvline(knl,ls=':',color='darkgrey')
        ax[1].axvline(sig, ls=':',color='darkgrey')

    # Tidy up the plot.

    ax[0].legend(ncol=2,framealpha=0.5, prop=fontmanage)
    ax[1].legend(ncol=2,framealpha=0.5, prop=fontmanage)
    ax[0].set_xlim(0.008,2.0)
    ax[1].set_xlim(0.7,30.0)
    ax[0].set_ylim(1.0,6.0)
    ax[1].set_ylim(1.0,6.0)
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1].set_xlabel(r'$r\quad [h^{-1}\,{\rm Mpc}]$', fontdict=font)
    ax[0].set_ylabel(r'$b(k)$', fontdict=font)
    ax[1].set_ylabel(r'$b(r)$', fontdict=font)
    # Put on some more labels.
    for axis in ax:
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'HI_bias.pdf')
    #





def make_clustering_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [2.0,4.0,6.0]

    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(9,6), sharex='col', sharey='col')

    for iz, zz in enumerate(zlist):

        aa = 1.0/(1.0+zz)
        # Put on a fake symbol for the legend.

        # Plot the data, Pmm, Phm, Phh
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        pkm = pkd[:,3]
        pkh = pkd[:,3] * pkd[:,2]**2
        pkx = pkd[:,3] * pkd[:,1]
        bb = pkd[1:6,1].mean()

        ax[iz,0].plot(pkd[:,0],pkm,'C0-', label=r'$P_{\rm m-m}$')
        ax[iz,0].plot(pkd[:,0],pkx,'C1-', label=r'$P_{\rm HI-m}$')
        ax[iz,0].plot(pkd[:,0],pkh,'C2-', label=r'$P_{\rm HI-HI}$')
        # and the linear theory counterparts.
        pk = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        ax[iz,0].plot(pk[:,0], 1**2*pk[:,1],'C0:')
        ax[iz,0].plot(pk[:,0],bb**1*pk[:,1],'C1:')
        ax[iz,0].plot(pk[:,0],bb**2*pk[:,1],'C2:')
        # put on a marker for knl.
        knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        ax[iz,0].axvline(knl,ls=':',color='darkgrey')
        ax[iz,0].text(1.1*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='left',va='center', fontdict=font)

        #
        xim = np.loadtxt(dpathxi + "ximatter_{:06.4f}.txt".format(aa))
        xix = np.loadtxt(dpathxi + "ximxh1mass_{:06.4f}.txt".format(aa))
        xih = np.loadtxt(dpathxi + "xih1mass_{:06.4f}.txt".format(aa))

        ax[iz,1].plot(xim[:,0],xim[:,1],'C0')
        ax[iz,1].plot(xix[:,0],xix[:,1],'C1-')
        ax[iz,1].plot(xih[:,0],xih[:,1],'C2-')
        sig = np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        ax[iz,1].axvline(sig,ls=':',color='darkgrey')
        ax[iz,1].text(1.1*sig,2e-2,r'$\Sigma_{\rm nl}$',color='darkgrey',\
                       ha='left',va='center', fontdict=font)

        ax[iz,1].text(12,2,r'$z=%.1f$'%zz,color='black',\
                       ha='left',va='center', fontdict=font)
    # Tidy up the plot.

    ax[0,0].legend(ncol=1,framealpha=0.5, prop=fontmanage)
    ax[0,0].set_xlim(0.008,2.0)
    ax[0,1].set_xlim(0.7,30.0)
    ax[0,0].set_ylim(3.0,3e4)
    ax[0,1].set_ylim(1e-2,10)

    # Put on some more labels.
    ax[2,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[2,1].set_xlabel(r'$r\quad [h^{-1}\,{\rm Mpc}]$', fontdict=font)
    for axis in ax[:,0]: axis.set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    for axis in ax[:,1]: axis.set_ylabel(r'$\xi(r)$', fontdict=font)
    # Put on some more labels.
    for axis in ax.flatten():
        axis.set_xscale('log')
        axis.set_yscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'HI_clustering.pdf')
    #


if __name__=="__main__":
    make_bias_plot()
    make_clustering_plot()
    #
