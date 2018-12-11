#!/usr/bin/env python3
#
# Plots the real-space power spectra for the HI.
#
import numpy as np
import matplotlib.pyplot as plt



# The directories where the data are stored.
scratch = '/global/cscratch1/sd/yfeng1/m3127/highres/10240-9100-fixed/'
project = '/project/projectdirs/m3127/H1mass/highres/2560-9100-fixed/'



def make_pkr_plot():
    """Does the work of making the real-space P(k) figure."""
    zlist = [2.0,3.0,4.0,5.0]
    blist = [1.8,2.2,2.6,3.0]
    # Now make the figure.
    fig,ax = plt.subplots(2,2,figsize=(6,4),sharex=True,sharey=True)
    for ix in range(ax.shape[0]):
        for iy in range(ax.shape[1]):
            ii = ix*ax.shape[1] + iy
            zz = zlist[ii]
            aa = 1.0/(1.0+zz)
            bb = blist[ii]
            # Plot the data, Pmm, Phm, Phh
            base= project+"fastpm_{:06.4f}/".format(aa)
            pkh = np.loadtxt(base+"pkh1mass.txt")
            ax[ix,iy].plot(pkh[1:,0],pkh[1:,1],'b-',alpha=0.75)
            pkx = np.loadtxt(base+"pkh1massxm.txt")
            ax[ix,iy].plot(pkx[1:,0],pkx[1:,1],'c-',alpha=0.75)
            pkm = np.loadtxt(scratch+"powerspec_{:06.4f}.txt".format(aa))
            pkm = np.loadtxt(base+"pkm.txt")
            ax[ix,iy].plot(pkm[1:,0],pkm[1:,1],'k-',alpha=0.75)
            # and the linear theory counterparts.
            pk = np.loadtxt("pklin_{:6.4f}.txt".format(aa))
            ax[ix,iy].plot(pk[:,0],bb**2*pk[:,1],'b:')
            ax[ix,iy].plot(pk[:,0],bb**1*pk[:,1],'c:')
            ax[ix,iy].plot(pk[:,0], 1**2*pk[:,1],'k:')
            # put on a marker for knl.
            knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
            ax[ix,iy].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
            ax[ix,iy].text(1.1*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                           ha='left',va='center')
            # Tidy up the plot.
            ax[ix,iy].set_xlim(0.02,2.0)
            ax[ix,iy].set_ylim(3.0,3e4)
            ax[ix,iy].set_xscale('log')
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].text(0.025,5.,"$z={:.1f}$".format(zz))
            #
    # Put on some more labels.
    ax[1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$')
    ax[1,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('HI_pkr.pdf')
    #





def make_bkr_plot():
    """Does the work of making the real-space b(k) figure."""
    zlist = [2.0,3.0,4.0,5.0,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,3.0),sharey=True)
    for ii,zz,col in zip([0,1,0,1,0],zlist,clist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        base= project+"fastpm_{:06.4f}/".format(aa)
        pkh = np.loadtxt(base+"pkh1mass.txt")
        pkx = np.loadtxt(base+"pkh1massxm.txt")
        pkm = np.loadtxt(base+"pkm.txt")
        bka = np.sqrt( pkh[1:,1]/pkm[1:,1] )
        bkx = pkx[1:,1]/pkm[1:,1]
        ax[ii].plot(pkh[1:,0],bka,col+'-',label="$z={:.0f}$".format(zz))
        ax[ii].plot(pkh[1:,0],bkx,col+':')
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend()
        ax[ii].set_xlim(0.02,2.0)
        ax[ii].set_ylim(1.0,5.0)
        ax[ii].set_xscale('log')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0].set_ylabel(r'$b(k)$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('HI_bkr.pdf')
    #







if __name__=="__main__":
    make_pkr_plot()
    make_bkr_plot()
    #
