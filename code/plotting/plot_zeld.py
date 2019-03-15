#!/usr/bin/env python3
#
# Plots the power spectra for the matter and HI,
# compared to the Zeldovich/ZEFT models.
#
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
args = parser.parse_args()
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256

figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass

db = '../../data/outputs/%s/ModelA/'%suff
tb = '../..//theory/'



def make_pkr_plot():
    """Does the work of making the real-space P(k) figure."""
    zlist = [2.000,6.000]
    #b1lst = [0.938,2.788]
    #b2lst = [0.400,5.788]
    #alpha = [1.663,0.138]
    b1lst = [0.900,2.750]
    b2lst = [0.800,6.250]
    alpha = [1.500,0.145]

    # Now make the figure.
    fig,ax = plt.subplots(2,2,figsize=(6,4),sharex=True,\
               gridspec_kw={'height_ratios':[3,1]})
    for ii in range(ax.shape[1]):
        zz = zlist[ii]
        aa = 1.0/(1.0+zz)
        bb = b1lst[ii] + 1.0
        b1 = b1lst[ii]
        b2 = b2lst[ii]
        # Compute knl
        pk  = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        plh = pk[:,1] * (1+b1lst[ii])**2
        plx = pk[:,1] * (1+b1lst[ii])
        plm = pk[:,1] *  1.0
        ax[0,ii].plot(pk[:,0],plm,'C0:', lw=1.5)
        ax[0,ii].plot(pk[:,0],plx,'C1:', lw=1.5)
        ax[0,ii].plot(pk[:,0],plh,'C2:', lw=1.5)
        # Plot the data, Pmm, Phm, Phh
        pkd = np.loadtxt(db+"HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        pkm = pkd[:,3]
        pkh = pkd[:,3] * pkd[:,2]**2
        pkx = pkd[:,3] * pkd[:,1]
        ax[0,ii].plot(pkd[:,0],pkm,'C0-',alpha=0.75, lw=1.2, label=r'$P_{\rm m-m}$')
        ax[0,ii].plot(pkd[:,0],pkx,'C1-',alpha=0.75, lw=1.2, label=r'$P_{\rm HI-m}$')
        ax[0,ii].plot(pkd[:,0],pkh,'C2-',alpha=0.75, lw=1.2, label=r'$P_{\rm HI-HI}$')
        # Now Zeldovich.
        #pkz = np.loadtxt("pkzel_{:6.4f}.txt".format(aa))
        pkz = np.loadtxt(tb+"zeld_{:6.4f}.pkr".format(aa))
        kk  = pkz[:,0]
        pzh = (1+alpha[ii]*kk**2)*pkz[:,1]+b1*pkz[:,2]+b2*pkz[:,3]+\
              b1**2*pkz[:,4]+b2**2*pkz[:,5]+b1*b2*pkz[:,6]
        pzx = (1+alpha[ii]*kk**2)*pkz[:,1]+0.5*b1*pkz[:,2]+0.5*b2*pkz[:,3]
        pzm = (1+alpha[ii]*kk**2)*pkz[:,1]
        ax[0,ii].plot(kk[kk<knl],pzm[kk<knl],'C0--', lw=2.2)
        ax[0,ii].plot(kk[kk<knl],pzx[kk<knl],'C1--', lw=2.2)
        ax[0,ii].plot(kk[kk<knl],pzh[kk<knl],'C2--', lw=2.2)
        # Now plot the ratios.
        ww = np.nonzero( pkd[:,0]<knl )[0]
        rh = np.interp(pkd[ww,0],kk,pzh)/pkh[ww]
        rx = np.interp(pkd[ww,0],kk,pzx)/pkx[ww]
        rm = np.interp(pkd[ww,0],kk,pzm)/pkm[ww]
        ax[1,ii].plot(pkd[ww,0],1/rm,'C0', lw=1.2)
        ax[1,ii].plot(pkd[ww,0],1/rx,'C1', lw=1.2)
        ax[1,ii].plot(pkd[ww,0],1/rh,'C2', lw=1.2)
        # Add a grey shaded region.
        ax[1,ii].fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
        ax[1,ii].fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
        # put on a line for knl.
        ax[0,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        ax[0,ii].text(0.9*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='right',va='center', fontdict=font)
        ax[1,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0,ii].set_xlim(0.02,1.0)
        ax[0,ii].set_ylim(2.0,3e4)
        ax[0,ii].set_xscale('log')
        ax[0,ii].set_yscale('log')
        ax[0,ii].text(0.025,150.,"$z={:.1f}$".format(zz), fontdict=font)
        ax[1,ii].set_xlim(0.02,1.0)
        ax[1,ii].set_ylim(0.90,1.1)
        ax[1,ii].set_xscale('log')
        ax[1,ii].set_yscale('linear')
        #
    # Suppress the y-axis labels on not-column-0.
    for ii in range(1,ax.shape[1]):
        ax[0,ii].get_yaxis().set_visible(False)
        ax[1,ii].get_yaxis().set_visible(False)
    # Put on some more labels.
    ax[0,0].legend(prop=fontmanage)
    ax[1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    ax[1,0].set_ylabel(r'$P_{N-body}/P_Z$', fontdict=font)

    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + '/zeld_pkr.pdf')
    #




# z= 2.0  knl= 0.4026048003353967  best= [0.925  1.     0.05   4.7625]  lnL= -0.002645548824930343
# z= 6.0  knl= 0.9269032745448027  best= [ 2.725   7.375  -1.375   2.2875]  lnL= -0.004759269527165592






def make_pks_plot():
    """Does the work of making the redshift-space P(k) figure."""
    zlist = [2.000,6.000]
    b1lst = [0.925,2.725]
    b2lst = [1.000,7.375]
    a0lst = [0.050,-1.38]
    #a2lst = [4.763,2.288]
    a2lst = [3.800,2.000]
    # Now make the figure.
    fig,ax = plt.subplots(2,2,figsize=(6,4),sharex=True,\
               gridspec_kw={'height_ratios':[3,1]})
    for ii in range(ax.shape[1]):
        zz = zlist[ii]
        aa = 1.0/(1.0+zz)
        bb = b1lst[ii] + 1.0
        b1 = b1lst[ii]
        b2 = b2lst[ii]
        a0 = a0lst[ii]
        a2 = a2lst[ii]
        # Compute knl.
        pk  = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        # Plot the data, monopole and quadrupole.
        pkd = np.loadtxt(db+"HI_pks_ll_{:06.4f}.txt".format(aa))
        pk0 = pkd[:,1]
        pk2 = pkd[:,2]
        ax[0,ii].plot(pkd[:,0],pk0,'C0-',alpha=0.75, lw=1.2, label=r'$\ell=0$')
        ax[0,ii].plot(pkd[:,0],pk2,'C1-',alpha=0.75, lw=1.2, label=r'$\ell=2$')
        # Now Zeldovich.
        pz0 = np.loadtxt(tb+"zeld_{:6.4f}.pk0".format(aa))
        pz2 = np.loadtxt(tb+"zeld_{:6.4f}.pk2".format(aa))
        kk  = pz0[:,0]
        pz0 = (1+a0*kk**2)*pz0[:,1]+b1*pz0[:,2]+b2*pz0[:,3]+\
              b1**2*pz0[:,4]+b2**2*pz0[:,5]+b1*b2*pz0[:,6]
        pz2 = (1+a2*kk**2)*pz2[:,1]+b1*pz2[:,2]+b2*pz2[:,3]+\
              b1**2*pz2[:,4]+b2**2*pz2[:,5]+b1*b2*pz2[:,6]
        ax[0,ii].plot(kk[kk<knl],pz0[kk<knl],'C0--', lw=2.2)
        ax[0,ii].plot(kk[kk<knl],pz2[kk<knl],'C1--', lw=2.2)
        # Now plot the ratios.
        ww = np.nonzero( pkd[:,0]<knl )[0]
        r0 = np.interp(pkd[ww,0],kk,pz0)/pk0[ww]
        r2 = np.interp(pkd[ww,0],kk,pz2)/pk2[ww]
        ax[1,ii].plot(pkd[ww,0],1/r0,'C0', lw=1.2)
        ax[1,ii].plot(pkd[ww,0],1/r2,'C1', lw=1.2)
        # Add a grey shaded region.
        ax[1,ii].fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
        ax[1,ii].fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
        # put on a line for knl.
        ax[0,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        ax[0,ii].text(0.9*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='right',va='center', fontdict=font)
        ax[1,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0,ii].set_xlim(0.02,1.0)
        ax[0,ii].set_ylim(2.0,3e4)
        ax[0,ii].set_xscale('log')
        ax[0,ii].set_yscale('log')
        ax[0,ii].text(0.025,100.,"$z={:.1f}$".format(zz), fontdict=font)
        ax[1,ii].set_xlim(0.02,1.0)
        ax[1,ii].set_ylim(0.90,1.1)
        ax[1,ii].set_xscale('log')
        ax[1,ii].set_yscale('linear')
        #
    # Suppress the y-axis labels on not-column-0.
    for ii in range(1,ax.shape[1]):
        ax[0,ii].get_yaxis().set_visible(False)
        ax[1,ii].get_yaxis().set_visible(False)
    # Put on some more labels.
    ax[0,0].legend(prop=fontmanage)
    ax[1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$P_\ell(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    ax[1,0].set_ylabel(r'$P_{N-body}/P_Z$', fontdict=font)

    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + '/zeld_pks.pdf')
    #plt.savefig('zeld_pks.pdf')
    #









if __name__=="__main__":
    make_pkr_plot()
    make_pks_plot()
    #
