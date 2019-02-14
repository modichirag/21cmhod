#!/usr/bin/env python3
#
# Plots the real-space correlation functions and biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt



# The directories where the data are stored.
scratch = '/global/cscratch1/sd/yfeng1/m3127/highres/10240-9100-fixed/'
project = '/project/projectdirs/m3127/H1mass/highres/2560-9100-fixed/'



def make_xib_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [2.0,4.0,6.0]
    blist = [1.8,2.6,3.5]
    clist = ['b','g','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,3.0))
    for zz,bb,col in zip(zlist,blist,clist):
        aa = 1.0/(1.0+zz)
        # Put on a fake symbol for the legend.
        ax[0].plot([100],[100],'s',color=col,label="$z={:.1f}$".format(zz))
        # Plot the data, xi_mm, xi_hm, xi_hh
        base= project+"fastpm_{:06.4f}/ss-1000/".format(aa)
        xih = np.loadtxt(base+"xih1mass.txt")
        #ax[0].plot(xih[:,0],xih[:,0]**2*xih[:,1],'s--',color=col,alpha=0.75)
        ax[0].plot(xih[:,0],xih[:,1],'s--',color=col,mfc='None',alpha=0.75)
        xix = np.loadtxt(base+"ximxh1mass.txt")
        #ax[0].plot(xix[:,0],xix[:,0]**2*xix[:,1],'d:',color=col,alpha=0.75)
        ax[0].plot(xix[:,0],xix[:,1],'d:',color=col,mfc='None',alpha=0.75)
        xim = np.loadtxt(base+"ximatter.txt")
        #ax[0].plot(xim[:,0],xim[:,0]**2*xim[:,1],'o-',color=col,alpha=0.75)
        ax[0].plot(xim[:,0],xim[:,1],'o-',color=col,mfc='None',alpha=0.75)
        # and the inferred biases.
        ba = np.sqrt(xih[:,1]/xim[:,1])
        bx = xix[:,1]/xim[:,1]
        ax[1].plot(xih[:,0],ba,'s--',color=col,mfc='None',alpha=0.75)
        ax[1].plot(xih[:,0],bx,'d:' ,color=col,mfc='None',alpha=0.75)
        # put on a line for Sigma -- labels make this too crowded.
        pk  = np.loadtxt("pklin_{:6.4f}.txt".format(aa))
        Sig = np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        ax[0].plot([Sig,Sig],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0].set_xlim(0.5,20.0)
        ax[0].set_ylim(0.01,20.)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        #
        ax[1].set_xlim(0.5,20.)
        ax[1].set_ylim(1.0,5.0)
        ax[1].set_xscale('log')
        ax[1].set_yscale('linear')
    ax[0].legend()
    # Put on some more labels.
    ax[0].set_xlabel(r'$r\quad [h^{-1}\,{\rm Mpc}]$')
    ax[1].set_xlabel(r'$r\quad [h^{-1}\,{\rm Mpc}]$')
    ax[0].set_ylabel(r'$\xi(r)$')
    ax[1].set_ylabel(r'Bias')
    # and finish up.
    plt.tight_layout()
    plt.savefig('HI_xib.pdf')
    #










if __name__=="__main__":
    make_xib_plot()
    #
