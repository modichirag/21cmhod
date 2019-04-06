#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
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

#
sys.path.append('../utils/')
sys.path.append('../')
from uvbg import qlf
import mymass_function as massfunc
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
omM =  cosmodef['omegam']
mf = massfunc.Mass_Func('../../data/pk_Planck2018BAO_matterpower_z000.dat', M=omM)


#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
args = parser.parse_args()
boxsize = args.size


##
suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256

figpath = '../../figs/%s/'%(suff)


model = 'ModelA'
dpath = '../../data/outputs/%s/%s/'%(suff, model)



def make_mf_plot(fname, fsize=13):
    """Plot halo mass function as a check for the code"""
    zlist = [2.0,4.0, 6.0]
    # Now make the figure.

    fig,ax = plt.subplots(1,2,figsize=(7, 3))
    ax = ax[::-1]

    for iz, zz in enumerate(zlist):
        aa  = 1.0/(1.0+zz)
        # Read the data from file.
        mlow, mhigh, mm, nn, mmh, nnh = np.loadtxt(dpath + "smf_{:06.4f}.txt".format(aa), unpack=True)
        h = cosmodef['h']
        diff = mhigh-mlow
        #mff = nn/(bs/h)**3/diff/np.log(10)
        mff = nn/(bs/h)**3/diff
        ax[0].plot(np.log10(mm), np.log10(mff) , 'C%do'%iz, label='$z = %.1f$'%zz, markersize=3)
        mm, nn = mmh, nnh
        #mff = nn/(bs/h)**3/diff/np.log(10)
        #ax[0].plot(np.log10(mm), np.log10(mff) , 'C%dx'%iz, label='$z = %.1f$'%zz)
        
        mlow, mhigh, mm, nn = np.loadtxt(dpath + "hmf_{:06.4f}.txt".format(aa), unpack=True)
        diff = mhigh-mlow
        mff = nn/bs**3/diff/np.log(10)
        ax[1].plot(np.log10(mm), np.log10(mff), 'C%do'%iz, label='$z = %.1f$'%zz, markersize=3)
        if iz == 2: ax[1].plot(np.log10(mm), np.log10(mf.STf(mm, aa)), 'C%d--'%iz, label="ST")
        ax[1].plot(np.log10(mm), np.log10(mf.STf(mm, aa)), 'C%d--'%iz)

#

    smfpath = '/global/u1/c/chmodi/Programs/21cm/21cmhod/data/behroozi-2013-data-compilation/smf_ms/'
    data = np.loadtxt(smfpath + '/mortlock_z2.0.smf').T
    ax[0].errorbar(data[0], data[1], [data[3], data[2]], color='C0', ls="--")
    data = np.loadtxt(smfpath + '/kslee_z4.smf').T
    ax[0].errorbar(data[0], data[1], [data[3], data[2]], color='C1', ls="--")
    data = np.loadtxt(smfpath + '/stark_z6.smf').T
    #data = np.loadtxt(smfpath + '/kslee_z5.smf').T
    ax[0].errorbar(data[0], data[1], [data[3], data[2]], color='C2', ls="--")


    #Formatting
    ax[1].legend(prop=fontmanage)
    ax[0].set_ylim(-6.5, -1.2)
    for axis in ax:
        #axis.set_xscale('log')
        #axis.set_yscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
    
    ax[0].set_xlabel(r'${\rm log M_*}(\rm M_{\odot})$', fontdict=font)
    ax[1].set_xlabel(r'${\rm log M}(\rm M_{\odot}/h)$', fontdict=font)
    #ax[0].set_ylabel(r'log ( SMF $=\, \frac{dn}{d{\rm log}M} {\rm Mpc}^{-3}{\rm dex}^{-1} $)', fontdict=font)
    #ax[1].set_ylabel(r'log ( HMF $=\, \frac{dn}{d{\rm ln}M} {\rm Mpc^{-3}h}^{3} $)', fontdict=font)
    ax[0].set_ylabel(r'log (SMF) [${\rm Mpc}^{-3}{\rm dex}^{-1} $]', fontdict=font)
    ax[1].set_ylabel(r'log (HMF) [${\rm Mpc^{-3}h}^{3} $]', fontdict=font)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



def make_quasar_plot(fname, fsize=13):
    """Plot halo mass function as a check for the code"""
    zlist = [3.5, 4.0, 6.0]
    # Now make the figure.

    fig,ax = plt.subplots(1,2,figsize=(7, 3))


    qlfdat = np.loadtxt('../../data/qlf.dat').T
    for iz, zz in enumerate(zlist):
        aa  = 1.0/(1.0+zz)
        # Read the data from file.
        mlow, mhigh, mm, nn = np.loadtxt(dpath + "qlf_{:06.4f}.txt".format(aa), unpack=True)
        h = cosmodef['h']
        diff = mhigh-mlow
        mff = nn/(bs/h)**3/diff
        ax[0].plot(mm, np.log10(mff) , 'C%d-'%iz, label='$z = %.1f$'%zz, markersize=3)
        ax[0].plot(mm, np.log10(qlf(mm, zz)) , 'C%d--'%iz, lw=0.5)
        mask = (qlfdat[5] > zz-0.2) & (qlfdat[5] < zz+0.2)
        ax[0].errorbar(qlfdat[6][mask], qlfdat[9][mask], qlfdat[[11,10]].T[mask].T, qlfdat[[7,8]].T[mask].T, fmt='.', color='C%d'%iz)
    
    if 'big' in suff: dpathbig = dpath
    else: dpathbig = '../../data/outputs/%s/%s/'%(suff+'-big', model)
    zz = 4
    aa = 1/(zz+1)
    lum = np.loadtxt(dpathbig + 'uvbg/Lum_bias_{:.4f}.txt'.format(aa)).T
    uv = np.loadtxt(dpathbig + 'uvbg/UVbg_bias_{:.4f}.txt'.format(aa)).T
    uv2 = np.loadtxt(dpathbig + 'uvbg/UVbg_star_bias_{:.4f}.txt'.format(aa)).T
    h1fid = np.loadtxt(dpathbig + 'HI_bias_{:.4f}.txt'.format(aa)).T
    h1uv = np.loadtxt(dpathbig + 'uvbg/HI_UVbg_ap2p5_bias_{:.4f}.txt'.format(aa)).T
    h1uv2 = np.loadtxt(dpathbig + 'uvbg/HI_UVbg_ap2p5_star_bias_{:.4f}.txt'.format(aa)).T
    k = lum[0]
    ax[1].plot(k, lum[2]**2*lum[3], label='L$_Q$')
    ax[1].plot(k, uv[2]**2*uv[3], label=r'$\Gamma_{\rm QSO}$')
    ax[1].plot(k, uv2[2]**2*uv2[3], label=r'$\Gamma_{\rm QSO+star}$')
    #ax[1].plot(k, h1uv[2]**2*h1uv[3], '--', label='HI: QSO')
    #ax[1].plot(k, h1uv2[2]**2*h1uv2[3], '--', label='HI: QSO+Stellar')
    #ax[1].plot(k, h1fid[2]**2*h1fid[3], '-', label='HI: fid')
    ax[1].loglog()

    ax[1].text(1e0 ,100,r'$z = %.1f$'%zz,color='k',\
                           ha='center',va='center', fontdict=font)


    #Formatting
    ax[0].legend(prop=fontmanage)
    ax[1].legend(ncol=1, loc=3, prop=fontmanage)

    ax[0].set_xlim(-5, -30)
    ax[0].set_ylim(-11, -2)
    ax[1].set_ylim(1e-2, 5e4)

    for axis in ax:
        #axis.set_xscale('log')
        #axis.set_yscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
    
    ax[0].set_xlabel(r'${\rm M_{1450}}$', fontdict=font)
    ax[1].set_xlabel(r' $k\,\, [h{\rm Mpc}^-1]$', fontdict=font)
    #ax[0].set_ylabel(r'log ( SMF $=\, \frac{dn}{d{\rm log}M} {\rm Mpc}^{-3}{\rm dex}^{-1} $)', fontdict=font)
    #ax[1].set_ylabel(r'log ( HMF $=\, \frac{dn}{d{\rm ln}M} {\rm Mpc^{-3}h}^{3} $)', fontdict=font)
    ax[0].set_ylabel(r'log ($\Phi$) [${\rm Mpc}^{-3}{\rm mag}^{-1} $]', fontdict=font)
    ax[1].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



if __name__=="__main__":
    make_mf_plot(figpath + 'MF.pdf')
    make_quasar_plot(figpath + 'QLF.pdf')
