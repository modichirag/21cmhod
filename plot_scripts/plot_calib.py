#!/usr/bin/env python3
#
# Plots OmHI vs. z, with observational data.
#
import numpy as np
import matplotlib.pyplot as plt






def make_calib_plot():
    """Does the work of making the calibration figure."""
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,2.5))
    # The left hand panel is the CDDF.
    ax[0].plot([0,1],[1,0])
    ax[0].set_xlabel(r'$N_{HI}\quad [{\rm cm}^{-2}]$')
    ax[0].set_ylabel(r'$N(>N_{HI})$')
    # The right hand panel is OmegaHI vs. z.
    # Read in the data and convert to "normal" OmegaHI convention.
    dd = np.loadtxt("omega_HI_obs.txt")
    Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
    ax[1].errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
                fmt='s',mfc='None')
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    ax[1].plot(zz,4e-4*(1+zz)**0.6/Ez**2,'k-')
    # Now plot the simulation points.
    dd = np.loadtxt("omega_HI_sim.txt")
    ax[1].plot(dd[:,0],dd[:,1],'md')
    # Tidy up the plot.
    ax[1].set_xlim(1,6)
    ax[1].set_ylim(4e-6,3e-4)
    ax[1].set_xscale('linear')
    ax[1].set_yscale('log')
    # Put on some more labels.
    ax[1].set_xlabel(r'$z$')
    ax[1].set_ylabel(r'$\Omega_{HI}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('calib.pdf')
    #











if __name__=="__main__":
    make_calib_plot()
    #
