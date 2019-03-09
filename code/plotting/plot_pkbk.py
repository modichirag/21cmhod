#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
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
parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-s', '--size', help='which box size simulation', default='small')
args = parser.parse_args()
if args.model == None:
    print('Specify a model name')
    sys.exit()
print(args, args.model)


model = args.model #'ModelD'
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass



def make_pkr_plot(fname):
    """Does the work of making the real-space P(k) figure."""
    zlist = [2.00,2.50,3.00,4.00,5.00,6.00]
    blist = [1.91,2.02,2.18,2.60,3.10,3.66]
    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(9,6),sharex=True,sharey=True)
    for ix in range(ax.shape[0]):
        for iy in range(ax.shape[1]):
            ii = ix*ax.shape[1] + iy
            zz = zlist[ii]
            aa = 1.0/(1.0+zz)
            bb = blist[ii]
            # Plot the data, Pmm, Phm, Phh
            pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
            pkm = pkd[:,3]
            pkh = pkd[:,3] * pkd[:,2]**2
            pkx = pkd[:,3] * pkd[:,1]
            ax[ix,iy].plot(pkd[:,0],pkh,'b-',alpha=0.75)
            ax[ix,iy].plot(pkd[:,0],pkx,'c-',alpha=0.75)
            ax[ix,iy].plot(pkd[:,0],pkm,'k-',alpha=0.75)
            # and the linear theory counterparts.
            pk = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
            ax[ix,iy].plot(pk[:,0],bb**2*pk[:,1],'b:')
            ax[ix,iy].plot(pk[:,0],bb**1*pk[:,1],'c:')
            ax[ix,iy].plot(pk[:,0], 1**2*pk[:,1],'k:')
            # put on a marker for knl.
            knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
            ax[ix,iy].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
            ax[ix,iy].text(1.1*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                           ha='left',va='center', fontdict=font)
            # Tidy up the plot.
            ax[ix,iy].set_xlim(0.02,2.0)
            ax[ix,iy].set_ylim(3.0,3e4)
            ax[ix,iy].set_xscale('log')
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].text(0.025,5.,"z={:.1f}".format(zz), fontdict=font)
            #
    # Put on some more labels.
    ax[2,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[2,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    ax[1,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    ax[2,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    # and finish up.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    plt.tight_layout()
    plt.savefig(fname)
    #




def make_pks_plot(fname):
    """Does the work of making the redshift-space P(k) figure."""
    zlist = [2.00,2.50,3.00,4.00,5.00,6.00]
    blist = [1.91,2.02,2.18,2.60,3.10,3.66]
    mcols = ['b','c','m','r']	# Colors for each mu bin.
    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(9,6),sharex=True,sharey=True)
    for ix in range(ax.shape[0]):
        for iy in range(ax.shape[1]):
            ii = ix*ax.shape[1] + iy
            zz = zlist[ii]
            aa = 1.0/(1.0+zz)
            bb = blist[ii]
            ff = (0.31/(0.31+0.69*aa**3))**0.55
            # Plot the data, Pmm, Phm, Phh
            pkd = np.loadtxt(dpath + "HI_pks_mu_{:06.4f}.txt".format(aa))[1:,:]
            for i in range(len(mcols)):
                mumin,mumax = float(i)/len(mcols),float(i+1)/len(mcols)
                lbl="${:.2f}<\\mu<{:.2f}$".format(mumin,mumax)
                if ix==0:
                    if (i%ax.shape[1]==iy):
                        ax[ix,iy].plot(pkd[:,0],pkd[:,0]**2*pkd[:,i+1],'-',\
                                       color=mcols[i],alpha=0.75,label=lbl)
                    else:
                        ax[ix,iy].plot(pkd[:,0],pkd[:,0]**2*pkd[:,i+1],'-',\
                                       color=mcols[i],alpha=0.75)
                else:
                    ax[ix,iy].plot(pkd[:,0],pkd[:,0]**2*pkd[:,i+1],'-',\
                                   color=mcols[i],alpha=0.75)
            # and the linear theory counterparts.
            pk = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
            for i in range(len(mcols)):
                mu = (i+0.5)/len(mcols)
                kf = (bb+ff*mu**2)**2
                ax[ix,iy].plot(pk[:,0],pk[:,0]**2*kf*pk[:,1],':',color=mcols[i])
            # put on a marker for knl.
            #knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
            #ax[ix,iy].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
            #ax[ix,iy].text(1.1*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
            #               ha='left',va='center')
            # Tidy up the plot.
            ax[ix,iy].set_xlim(0.02,2.0)
            ax[ix,iy].set_ylim(10.0,150.)
            ax[ix,iy].set_xscale('log')
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].text(0.025,110,"z={:.1f}".format(zz),ha='left',va='top', fontdict=font)
            #
            if ix==0:
                ax[ix,iy].legend(framealpha=0.5, prop=fontmanage)
    # Put on some more labels.
    ax[2,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[2,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$k^2\ P(k,\mu)$', fontdict=font)
    ax[1,0].set_ylabel(r'$k^2\ P(k,\mu)$', fontdict=font)
    ax[2,0].set_ylabel(r'$k^2\ P(k,\mu)$', fontdict=font)
    # and finish up.
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    plt.tight_layout()
    plt.savefig(fname)
    #







def make_bkr_plot(fname):
    """Does the work of making the real-space b(k) figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','b','c','c','g','g','m','r','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(7,3.5),sharey=True)
    ii = 0
    for zz,col in zip(zlist,clist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))
        bka = pkd[:,2]
        bkx = pkd[:,1]
        ax[ii].plot(pkd[:,0],bka,col+'-',label="z={:.1f}".format(zz))
        ax[ii].plot(pkd[:,0],bkx,col+':')
        ii = (ii+1)%2
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5, prop=fontmanage)
        ax[ii].set_xlim(0.02,2.0)
        ax[ii].set_ylim(1.0,6.0)
        ax[ii].set_xscale('log')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0].set_ylabel(r'$b(k)$', fontdict=font)
    for axis in ax:
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



if __name__=="__main__":
    make_pkr_plot(figpath + 'HI_pkr_%s.pdf'%model)
    make_pks_plot(figpath + 'HI_pks_%s.pdf'%model)
    make_bkr_plot(figpath + 'HI_bkr_%s.pdf'%model)
    #
