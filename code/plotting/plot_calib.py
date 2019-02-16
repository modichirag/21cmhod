#!/usr/bin/env python3
#
# Plots OmHI vs. z, with observational data.
#
import numpy as np
import matplotlib.pyplot as plt
from   scipy.interpolate import InterpolatedUnivariateSpline as Spline






def make_calib_plot():
    """Does the work of making the calibration figure."""
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,2.5))
    # The left hand panel is DLA bias vs. redshift.
    bDLA = np.loadtxt("boss_bDLA.txt")
    ax[0].errorbar(bDLA[:,0],bDLA[:,1],yerr=bDLA[:,2],fmt='o')
    ax[0].fill_between([1.5,3.5],[1.99-0.11,1.99-0.11],\
                                 [1.99+0.11,1.99+0.11],\
                       color='lightgrey',alpha=0.5)
    # The N-body results.
    bb = np.loadtxt("HI_bias_vs_z_fid.txt")
    ax[0].plot(bb[:,0],bb[:,2],'md')
    ss = Spline(bb[::-1,0],bb[::-1,2])
    ax[0].plot(np.linspace(1.5,3.5,100),ss(np.linspace(1.5,3.5,100)),'m--')
    #
    ax[0].set_xlabel(r'$z$')
    ax[0].set_ylabel(r'$b_{DLA}(z)$')
    # Tidy up.
    ax[0].set_xlim(1.95,3.25)
    ax[0].set_ylim(1,3)
    ax[0].set_xscale('linear')
    ax[0].set_yscale('linear')
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
    ######################dd[:,1] *= 3e5*(1+(3.5/dd[:,0])**6) / 2e9
    ax[1].plot(dd[:,0],dd[:,1],'md')
    ss = Spline(dd[:,0],dd[:,1])
    ax[1].plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'m--')
    # Tidy up the plot.
    ax[1].set_xlim(1,6.25)
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
