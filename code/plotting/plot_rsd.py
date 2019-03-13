#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
#
#
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

from nbodykit.cosmology.cosmology import Cosmology
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
dgrow = cosmo.scale_independent_growth_factor

#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='which box size simulation', default='small')
args = parser.parse_args()
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256
#
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except Exception as e: print(e)

models  = ['ModelA', 'ModelB', 'ModelC']






def make_pks_plot(fname, fsize=11):
    """Does the work of making the distribution figure."""

    # Now make the figure.

    ncols = 4
    zlist = [2.0,4.0,6.0]
    blist = []
    for iz, zz in enumerate(zlist):
        aa  = 1.0/(1.0+zz)
        bias = np.loadtxt('../../data/outputs/%s/ModelA/'%(suff) + "HI_bias_{:06.4f}.txt".format(aa))
        blist.append(bias[1:6, 1].mean())


    # Now make the figure.
    fig,ax = plt.subplots(3,4,figsize=(12,8),sharex=True,sharey=True)
    #for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):
    for im, model in enumerate(models):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        
        for ix in range(ax.shape[0]):
            zz = zlist[ix]
            aa = 1.0/(1.0+zz)
            pkd = np.loadtxt(dpath + "HI_pks_mu_{:06.4f}.txt".format(aa))[1:,:]
            bb = blist[ix]
            ff = (0.31/(0.31+0.69*aa**3))**0.55

            for iy in range(ax.shape[1]):

                mumin,mumax = float(iy)/ncols,float(iy+1)/ncols
                title="${:.2f}<\\mu<{:.2f}$".format(mumin,mumax)
                #if ix == 0: ax[ix, iy].set_title(title)
                ax[ix,iy].plot(pkd[:,0],pkd[:,0]**2*pkd[:,iy+1],'-',\
                                       color='C%d'%im,alpha=0.75,label=model)
                # and the linear theory counterparts.
                pk = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
                mu = (iy+0.5)/ncols
                kf = (bb+ff*mu**2)**2
                ax[ix,iy].plot(pk[:,0],pk[:,0]**2*kf*pk[:,1],':',color='k')
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
                text = "$z={:.1f}$".format(zz)
                text = text + '\n%s'%title
                ax[ix,iy].text(0.025,110,text,ha='left',va='top')
                #
                if ix==2 and iy==3:
                    ax[ix,iy].legend(framealpha=0.5, ncol=2, loc='lower right')
                    
    # Put on some more labels.
    for axis in ax[-1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    for axis in ax[:, 0]: axis.set_ylabel(r'$k^2\ P(k,\mu)$')
    for axis in ax.flatten(): 
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
            
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #




def make_pkmu_ratio_plot(fname, fsize=11):
    """Plot ratio of mu=1 to mu=0 wedge"""


    # Now make the figure.
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]

    fig,ax = plt.subplots(2, 3,figsize=(9,6),sharex=True,sharey=True)

    # Now make the figure.
    #for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):
    for im, model in enumerate(models):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        
        blist = []

        for iz, zz in enumerate(zlist):

            axis = ax.flatten()[iz]
            zz = zlist[iz]
            aa = 1.0/(1.0+zz)
            ff = (0.31/(0.31+0.69*aa**3))**0.55

            # and the linear theory counterparts.
            pkd = np.loadtxt(dpath + "HI_pks_mu_{:06.4f}.txt".format(aa))[1:,:]
            axis.plot(pkd[:,0],pkd[:,-1]/pkd[:,1],'-',\
                                       color='C%d'%im,alpha=0.95,label=model, lw=1.5)
                
        
            axis.set_xlim(0.02,2)
            axis.set_ylim(0.55, 3)
            axis.set_xscale('log')
            #axis.set_yscale('log')
            text = "$z={:.1f}$".format(zz)
            axis.text(0.025,1.4,text,ha='left',va='top')
            #
            if iz==0: axis.legend(framealpha=0.5, ncol=1, loc='lower left')
                    
    # Put on some more labels.
    for axis in ax[-1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    for axis in ax[:, 0]: axis.set_ylabel(r'$P(k, 0.87)/P(k, 0.12)$')
    for axis in ax.flatten(): 
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
            
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #

    #


def make_pkmudiff_ratio_plot(fname, fsize=11):
    """Plot ratio of mu=1 to mu=0 wedge"""


    # Now make the figure.
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]

    fig,ax = plt.subplots(2, 3,figsize=(9,6),sharex=True,sharey=True)

    # Now make the figure.
    #for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):
    for im, model in enumerate(models):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        
        blist = []

        for iz, zz in enumerate(zlist):

            axis = ax.flatten()[iz]
            zz = zlist[iz]
            aa = 1.0/(1.0+zz)
            ff = (0.31/(0.31+0.69*aa**3))**0.55

            # and the linear theory counterparts.
            klin, plin = np.loadtxt('../../data/pk_Planck2018BAO_matterpower_z000.dat', unpack=True)
            pkd = np.loadtxt(dpath + "HI_pks_mu_{:06.4f}.txt".format(aa))[1:,:]
            pkb = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
            bh = pkb[:,2]
            bb = bh[1:6].mean()
            mu = 1-0.5/4
            beta = ff/bb
            #diff = pkd[:, -1]-pkd[:, 1]
            diff = pkd[:, -1]-bh**2 *pkb[:,-1]

            #bpk = ius(pkb[:, 0], bb**2*pkb[:, -1])(pkd[:, 0])
            #axis.plot(pkd[:,0], (diff/bpk+1) /(1+beta*mu**2)**2 ,'-',\
            #                           color='C%d'%im,alpha=0.95,label=model, lw=1)

            bpk = ius(klin, bb**2*plin*dgrow(zz)**2)(pkd[:, 0])
            axis.plot(pkd[:,0], (diff/bpk+1) /(1+beta*mu**2)**2 ,'-',\
                                       color='C%d'%im,alpha=0.95,label=model, lw=2)

            bpk = ius(pkb[:, 0], bh**2*pkb[:, -1])(pkd[:, 0])
            if im:
                axis.plot(pkd[:,0], (diff/bpk+1) /(1+beta*mu**2)**2 ,'--',\
                                       color='C%d'%im,alpha=0.7, lw=2.)
            else:
                axis.plot(pkd[:,0], (diff/bpk+1) /(1+beta*mu**2)**2 ,'--',\
                                       color='C%d'%im,alpha=0.7,label='b(k)P(k)', lw=2.)
            
        
            axis.set_xlim(0.02,2)
            axis.set_ylim(0.55, 2)
            axis.set_xscale('log')
            #axis.set_yscale('log')
            text = "$z={:.1f}$".format(zz)
            axis.text(0.025,1.4,text,ha='left',va='top')
            #
            if iz==0: axis.legend(framealpha=0.5, ncol=1, loc='upper left')
                    
    # Put on some more labels.
    for axis in ax[-1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    for axis in ax[:, 0]: axis.set_ylabel(r'$P(k, 0.87)/P(k, 0.12)$')
    for axis in ax.flatten(): 
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
            
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #

    #


def make_pkll_ratio_plot(fname, fsize=11):
    """Does the work of making the distribution figure."""

    # Now make the figure.
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]

    fig,ax = plt.subplots(2, 3,figsize=(9,6),sharex=True,sharey=True)

    # Now make the figure.
    #for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):
    for im, model in enumerate(models):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        
        blist = []

        for iz, zz in enumerate(zlist):

            axis = ax.flatten()[iz]
            zz = zlist[iz]
            aa = 1.0/(1.0+zz)
            ff = (0.31/(0.31+0.69*aa**3))**0.55

            pkd = np.loadtxt(dpath + "HI_pks_ll_{:06.4f}.txt".format(aa))[1:,:]
            #pkd1 = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
            pkb = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
            bh = pkb[:,2]
            bb = bh[1:6].mean()
            
            pkr = bh**2 * pkb[:, 3]
            ipkr = ius(pkb[:, 0], pkr)
            axis.plot(pkd[:,0],pkd[:,1]/ipkr(pkd[:,0]),'-',\
                                       color='C%d'%im,alpha=0.9,label=model, lw=2)

            pkr = bb**2 * pkb[:, 3]
            ipkr = ius(pkb[:, 0], pkr)
            #axis.plot(pkd[:,0],pkd[:,1]/ipkr(pkd[:,0]),'--',\
            #                           color='C%d'%im,alpha=0.5,label=model, lw=2.5)
               
        
            axis.set_xlim(0.02,1.5)
            axis.set_ylim(0.9, 2.2)
            axis.set_xscale('log')
            #axis.set_yscale('log')
            text = "$z={:.1f}$".format(zz)
            axis.text(0.025,1.4,text,ha='left',va='top')
            #
            if iz==0: axis.legend(framealpha=0.5, ncol=1, loc='lower left')
                    
    # Put on some more labels.
    for axis in ax[-1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    for axis in ax[:, 0]: axis.set_ylabel(r'$P(k)_s/P(k)_r$')
    for axis in ax.flatten(): 
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
            
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #





def make_kaiser_ratio_plot(fname, fsize=11):
    """Does the work of making the distribution figure."""

    # Now make the figure.
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]

    fig,ax = plt.subplots(2, 3,figsize=(9,6),sharex=True,sharey=True)

    # Now make the figure.
    #for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):
    for im, model in enumerate(models):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        
        blist = []

        for iz, zz in enumerate(zlist):

            axis = ax.flatten()[iz]
            zz = zlist[iz]
            aa = 1.0/(1.0+zz)
            ff = (0.31/(0.31+0.69*aa**3))**0.55

            pkd = np.loadtxt(dpath + "HI_pks_ll_{:06.4f}.txt".format(aa))[1:,:]
            #pkd = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
            pkb = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
            bh = pkb[:,2]
            bb = bh[1:6].mean()
            #print(bh[1:6], bb)

            pkr = (bh**2 + 2*bh*ff/3 + ff**2/5)* pkb[:, 3]
            ipkr = ius(pkb[:, 0], pkr)
            #axis.plot(pkd[:,0],pkd[:,1]/ipkr(pkd[:,0]),'-',\
            #                           color='C%d'%im,alpha=0.95,label=model, lw=1.5)

            #pkr = (bb**2 + 2*bb*ff/3 + ff**2/5)* pkb[:, 3]
            pkr = bh**2*(1 + 2*ff/bb/3 + ff**2/bb**2/5)* pkb[:, 3]
            ipkr = ius(pkb[:, 0], pkr)
            axis.plot(pkd[:,0],pkd[:,1]/ipkr(pkd[:,0]),'-',\
                                       color='C%d'%im,alpha=0.9,label=model, lw=2)
               
        
            axis.set_xlim(0.02,2.0)
            axis.set_ylim(0.6, 1.4)
            axis.set_xscale('log')
            #axis.set_yscale('log')
            text = "$z={:.1f}$".format(zz)
            axis.text(0.025,1.2,text,ha='left',va='top')
            #
            if iz==0: axis.legend(framealpha=0.5, ncol=1, loc='lower left')
                    
    # Put on some more labels.
    for axis in ax[-1, :]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    for axis in ax[:, 0]: axis.set_ylabel(r'$P(k)_s/P(k)_r$')
    for axis in ax.flatten(): 
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontsize(fsize)
            
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



if __name__=="__main__":
    make_pkmudiff_ratio_plot(figpath + 'mudiffratio.pdf')
    make_pkmu_ratio_plot(figpath + 'muratio.pdf')
    make_pkll_ratio_plot(figpath + 'llratio.pdf')
    make_kaiser_ratio_plot(figpath + 'kaiserratio.pdf')
    make_pks_plot(figpath + 'pkmu.pdf')
#    #
