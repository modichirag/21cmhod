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
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
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





def make_redclustering_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [2.0,4.0,6.0]

    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(9,6), sharex='col', sharey='col')

    nmu = 4
    nell = 3
    
    for iz, zz in enumerate(zlist):

        aa = 1.0/(1.0+zz)
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        bb = pkd[1:6,1].mean()
        ff = (0.31/(0.31+0.69*aa**3))**0.55

        pkd = np.loadtxt(dpath + "HI_pks_mu_{:06.4f}.txt".format(aa))[1:,:]
        pk = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        for i in range(nmu):
            mumin,mumax = float(i)/nmu,float(i+1)/nmu
            lbl = None
            if i//2+1==iz: lbl="${:.2f}<\\mu<{:.2f}$".format(mumin,mumax)
            ax[iz,0].plot(pkd[:,0],pkd[:,0]**2*pkd[:,i+1],'C%d-'%i,\
                                   alpha=0.75,label=lbl)

            mu = (i+0.5)/nmu
            kf = (bb+ff*mu**2)**2
            ax[iz,0].plot(pk[:,0],pk[:,0]**2*kf*pk[:,1],'C%d:'%i)


        ##
        pkd = np.loadtxt(dpath + "HI_pks_ll_{:06.4f}.txt".format(aa))[1:,:]
        muu = np.linspace(-1, 1, 1000)
        kf = (bb+ff*muu**2)**2
        l0 = np.trapz(kf, muu) *1/2.
        l2 = np.trapz(kf* 1*(3*muu**2-1)/2., muu)*5/2.
        l4 = np.trapz(kf* 1*(35*muu**4-30*muu**2+3)/8., muu)*9/2.
        #nl0 = np.trapz(1+muu*0, muu) *1/2.
        #nl2 = np.trapz(1* (1*(3*muu**2-1)/2.)**2, muu) * 5/2.
        #nl4 = np.trapz(1* (1*(35*muu**4-30*muu**2+3)/8.)**2, muu) *9/2.
        #print(nl0, nl2, nl4)
        llfac = [l0,l2,l4]
        for i in range(nell):
            lbl = None
            if iz==i: lbl = r'$\ell=%d$'%(2*i)
            ax[iz,1].plot(pkd[:,0],pkd[:,0]**2*pkd[:,i+1],'C%d-'%i,\
                                   alpha=0.75,label=lbl)

            ax[iz,1].plot(pk[:,0],pk[:,0]**2*llfac[i]*pk[:,1],'C%d:'%i)


        ax[iz,0].text(2e-2, 100, r'$z=%.1f$'%zz,color='black',\
                       ha='left',va='center', fontdict=font)
    # Tidy up the plot.

    for axis in ax[:,0]: axis.legend(ncol=1,framealpha=0.5, prop=fontmanage)
    for axis in ax[:,1]: axis.legend(ncol=1,framealpha=0.5, prop=fontmanage, loc=2)
    ax[0,0].set_xlim(1e-2,2)
    ax[0,1].set_xlim(5e-2,2)
    ax[0,0].set_ylim(1e0,5e2)
    ax[0,1].set_ylim(5e-1,5e2)
#
    # Put on some more labels.
    for axis in ax[2]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in ax[:,0]: axis.set_ylabel(r'$k^2\ P(k,\mu)$', fontdict=font)
    for axis in ax[:,1]: axis.set_ylabel(r'$k^2P_\ell(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)

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
    plt.savefig(figpath + 'HI_redclustering.pdf')
    #


if __name__=="__main__":
    make_redclustering_plot()
    #
