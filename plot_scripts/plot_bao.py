#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.signal import savgol_filter











def make_bao_plot():
    """Does the work of making the BAO figure."""
    zlist = [2.0,3.0,4.0,5.0,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,3.0),sharey=True)
    ii,jj  = 0,0
    for zz,col in zip(zlist,clist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        pkd = np.loadtxt("HI_pks_1d_{:06.4f}.txt".format(aa))
        # Now read linear theory and put it on the same grid -- currently
        # not accounting for finite bin width.
        lin = np.loadtxt("pklin_{:06.4f}.txt".format(aa))
        lin = np.interp(pkd[:,0],lin[:,0],lin[:,1])
        # Take out the broad band.
        if False: # Use smoothing spline as broad-band/no-wiggle.
            knots=np.arange(0.05,0.5,0.05)
            ss  = Spline(pkd[:,0],pkd[:,1],t=knots)
            rat = pkd[:,1]/ss(pkd[:,0])
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ss  = savgol_filter(pkd[:,1],7,polyorder=2)
            rat = pkd[:,1]/ss
        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),col+'-',\
                    label="$z={:.1f}$".format(zz))
        if False: # Use smoothing spline as broad-band/no-wiggle.
            ss  = Spline(pkd[:,0],lin,t=knots)
            rat = lin/ss(pkd[:,0])
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ss  = savgol_filter(lin,7,polyorder=2)
            rat = lin/ss
        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),col+':')
        ii = (ii+1)%2
        jj =  jj+1
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5)
        ax[ii].set_xlim(0.05,0.4)
        ax[ii].set_ylim(0.75,1.5)
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0].set_ylabel(r'$P(k)/P_{\rm nw}(k)$+offset')
    # and finish up.
    plt.tight_layout()
    plt.savefig('HI_bao.pdf')
    #







if __name__=="__main__":
    make_bao_plot()
    #
