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

#
#
bs = 1024
suff = 'm1_00p3mh-alpha-0p8-subvol'
if bs == 1024: suff = suff + '-big'
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except Exception as e: print(e)



def make_omHI_plot(fname, fsize=12):
    """Does the work of making the distribution figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,axis = plt.subplots(figsize=(6, 5))

   # Read in the data and convert to "normal" OmegaHI convention.
    dd = np.loadtxt("../../data/omega_HI_obs.txt")
    Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
    axis.errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
                fmt='s',mfc='None')
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    axis.plot(zz,4e-4*(1+zz)**0.6/Ez**2,'k-')

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)

        omHI = np.loadtxt(dpath + "OmHI.txt")
        #omHI[:, 1] /= 10
        axis.plot(omHI[:, 0], omHI[:, 1], 'C%do'%im, label=model)

        ss = ius(omHI[::-1, 0], omHI[::-1, 1])
        axis.plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'C%d'%im)

    axis.set_yscale('log')
    axis.legend(fontsize=fsize)
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    axis.set_xlabel(r'$z$')
    axis.set_ylabel(r'$\Omega_{HI}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #







def make_omHI_plot_2(fname, fsize=12):
    """Does the work of making the distribution figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,axis = plt.subplots(figsize=(6, 5))

   # Read in the data and convert to "normal" OmegaHI convention.
    dd = np.loadtxt("../../data/omega_HI_obs.txt")
    #Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
    #axis.errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
    #            fmt='s',mfc='None')
    axis.errorbar(dd[:,0],1e-3*dd[:,1],yerr=1e-3*dd[:,2],fmt='s',mfc='None')
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    axis.plot(zz,4e-4*(1+zz)**0.6,'k-')

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        omz = []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            omHI = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa)).T
            omHI = (omHI[1]*omHI[2]).sum()/bs**3/27.754e10
            omHI *= (1+zz)**3
            if iz == 0: axis.plot(zz, omHI, 'C%do'%im, label=model)
            else: axis.plot(zz, omHI, 'C%do'%im)
            omz.append(omHI)

        ss = ius(zlist, omz)
        axis.plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'C%d'%im)

    axis.set_yscale('log')
    axis.legend(fontsize=fsize)
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    axis.set_xlabel(r'$z$')
    axis.set_ylabel(r'$\Omega_{HI}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #




def make_bias_plot(fname, fsize=12):
    """Does the work of making the distribution figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,axis = plt.subplots(figsize=(6, 5))

    bDLA = np.loadtxt("../../data/boss_bDLA.txt")
    axis.errorbar(bDLA[:,0],bDLA[:,1],yerr=bDLA[:,2],fmt='s', label='BOSS', color='m')
    axis.fill_between([1.5,3.5],[1.99-0.11,1.99-0.11],\
                                 [1.99+0.11,1.99+0.11],\
                       color='lightgrey',alpha=0.5)

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)

        bbs = []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            bias = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))
            bb = bias[1:10, 1].mean()
            bbs.append(bb)
            if iz == 0: 
                axis.plot(zz, bb, 'C%d'%im, marker='o', label=model)
            else:axis.plot(zz, bb, 'C%d'%im, marker='o')

        ss = ius(zlist,bbs)
        axis.plot(np.linspace(1.5,6.5,100),ss(np.linspace(1.5,6.5,100)),'C%d'%im)

    axis.legend(fontsize=fsize)
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    axis.set_xlabel(r'$z$')
    axis.set_ylabel(r'$b_{DLA}(z)$')
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #




def make_pks_plot(fname, fsize=11):
    """Does the work of making the distribution figure."""

    # Now make the figure.

    ncols = 4
    zlist = [2.0,4.0,6.0]
    blist = []
    for iz, zz in enumerate(zlist):
        aa  = 1.0/(1.0+zz)
        bias = np.loadtxt('../../data/outputs/%s/ModelA/'%(suff) + "HI_bias_{:06.4f}.txt".format(aa))
        blist.append(bias[1:10, 1].mean())


    # Now make the figure.
    fig,ax = plt.subplots(3,4,figsize=(12,8),sharex=True,sharey=True)
    for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):

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



def make_pkll_ratio_plot(fname, fsize=11):
    """Does the work of making the distribution figure."""

    # Now make the figure.
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]
    fig,ax = plt.subplots(2, 3,figsize=(9,6),sharex=True,sharey=True)

    # Now make the figure.
    for im, model in enumerate(['ModelA',  'ModelB', 'ModelC']):

        dpath = '../../data/outputs/%s/%s/'%(suff, model)        

        for iz, zz in enumerate(zlist):

            zz = zlist[iz]
            aa = 1.0/(1.0+zz)
            pkd = np.loadtxt(dpath + "HI_pks_ll_{:06.4f}.txt".format(aa))[1:,:]
            #pkd1 = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
            pkb = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
            pkr = pkb[:, 2]**2 * pkb[:, 3]
            ipkr = ius(pkb[:, 0], pkr)
            print(pkd.shape, pkb.shape)

            axis = ax.flatten()[iz]
            axis.plot(pkd[:,0],pkd[:,1]/ipkr(pkd[:,0]),'-',\
                                       color='C%d'%im,alpha=0.95,label=model, lw=2)
            #axis.plot(pkd1[:,0],pkd1[:,1]/ipkr(pkd1[:,0]),'--',\
            #                           color='C%d'%im,alpha=0.5,label=model)
                # and the linear theory counterparts.
            
        
            axis.set_xlim(0.02,1.3)
            axis.set_ylim(1.0, 1.5)
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



if __name__=="__main__":
    make_pkll_ratio_plot(figpath + 'monopoleratio.pdf')
    make_bias_plot(figpath + 'bias.pdf')
    make_omHI_plot_2(figpath + 'omHI.pdf')
    make_pks_plot(figpath + 'pkmu.pdf')
#    #
