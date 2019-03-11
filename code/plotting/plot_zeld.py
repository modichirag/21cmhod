#!/usr/bin/env python3
#
# Plots the power spectra for the matter and HI,
# compared to the Zeldovich/ZEFT models.
#
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt


db = '../../data/outputs/m1_00p3mh-alpha-0p8-subvol-big/ModelA/'
tb = '../..//theory/'








def make_pkr_plot():
    """Does the work of making the real-space P(k) figure."""
    zlist = [2.000,6.000]
    b1lst = [0.938,2.788]
    b2lst = [0.400,5.788]
    alpha = [1.663,0.138]
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
        ax[0,ii].plot(pk[:,0],plh,'b-')
        ax[0,ii].plot(pk[:,0],plx,'c-')
        ax[0,ii].plot(pk[:,0],plm,'k-')
        # Plot the data, Pmm, Phm, Phh
        pkd = np.loadtxt(db+"HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        pkm = pkd[:,3]
        pkh = pkd[:,3] * pkd[:,2]**2
        pkx = pkd[:,3] * pkd[:,1]
        ax[0,ii].plot(pkd[:,0],pkh,'b--',alpha=0.75)
        ax[0,ii].plot(pkd[:,0],pkx,'c--',alpha=0.75)
        ax[0,ii].plot(pkd[:,0],pkm,'k--',alpha=0.75)
        # Now Zeldovich.
        #pkz = np.loadtxt("pkzel_{:6.4f}.txt".format(aa))
        pkz = np.loadtxt(tb+"zeld_{:6.4f}.pkr".format(aa))
        kk  = pkz[:,0]
        pzh = (1+alpha[ii]*kk**2)*pkz[:,1]+b1*pkz[:,2]+b2*pkz[:,3]+\
              b1**2*pkz[:,4]+b2**2*pkz[:,5]+b1*b2*pkz[:,6]
        pzx = (1+alpha[ii]*kk**2)*pkz[:,1]+0.5*b1*pkz[:,2]+0.5*b2*pkz[:,3]
        pzm = (1+alpha[ii]*kk**2)*pkz[:,1]
        ax[0,ii].plot(kk[kk<knl],pzh[kk<knl],'b:')
        ax[0,ii].plot(kk[kk<knl],pzx[kk<knl],'c:')
        ax[0,ii].plot(kk[kk<knl],pzm[kk<knl],'k:')
        # Now plot the ratios.
        ww = np.nonzero( pkd[:,0]<knl )[0]
        rh = np.interp(pkd[ww,0],kk,pzh)/pkh[ww]
        rx = np.interp(pkd[ww,0],kk,pzx)/pkx[ww]
        rm = np.interp(pkd[ww,0],kk,pzm)/pkm[ww]
        ax[1,ii].plot(pkd[ww,0],rh,'b:')
        ax[1,ii].plot(pkd[ww,0],rx,'c:')
        ax[1,ii].plot(pkd[ww,0],rm,'k:')
        # Add a grey shaded region.
        ax[1,ii].fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
        ax[1,ii].fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
        # put on a line for knl.
        ax[0,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        ax[0,ii].text(0.9*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='right',va='center')
        ax[1,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0,ii].set_xlim(0.02,1.0)
        ax[0,ii].set_ylim(2.0,3e4)
        ax[0,ii].set_xscale('log')
        ax[0,ii].set_yscale('log')
        ax[0,ii].text(0.025,5.,"$z={:.1f}$".format(zz))
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
    ax[1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0,0].set_ylabel(r'$P(k)\quad [h^{-3}{\rm Mpc}^3]$')
    ax[1,0].set_ylabel(r'$P_Z/P_{N-body}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('zeld_pkr.pdf')
    #




# z= 2.0  knl= 0.4026048003353967  best= [0.925  1.     0.05   4.7625]  lnL= -0.002645548824930343
# z= 6.0  knl= 0.9269032745448027  best= [ 2.725   7.375  -1.375   2.2875]  lnL= -0.004759269527165592






def make_pks_plot():
    """Does the work of making the redshift-space P(k) figure."""
    zlist = [2.000,6.000]
    b1lst = [0.925,2.725]
    b2lst = [1.000,7.375]
    a0lst = [0.050,-1.38]
    a2lst = [4.763,2.288]
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
        ax[0,ii].plot(pkd[:,0],pk0,'b--',alpha=0.75)
        ax[0,ii].plot(pkd[:,0],pk2,'r--',alpha=0.75)
        # Now Zeldovich.
        pz0 = np.loadtxt(tb+"zeld_{:6.4f}.pk0".format(aa))
        pz2 = np.loadtxt(tb+"zeld_{:6.4f}.pk2".format(aa))
        kk  = pz0[:,0]
        pz0 = (1+a0*kk**2)*pz0[:,1]+b1*pz0[:,2]+b2*pz0[:,3]+\
              b1**2*pz0[:,4]+b2**2*pz0[:,5]+b1*b2*pz0[:,6]
        pz2 = (1+a2*kk**2)*pz2[:,1]+b1*pz2[:,2]+b2*pz2[:,3]+\
              b1**2*pz2[:,4]+b2**2*pz2[:,5]+b1*b2*pz2[:,6]
        ax[0,ii].plot(kk[kk<knl],pz0[kk<knl],'b:')
        ax[0,ii].plot(kk[kk<knl],pz2[kk<knl],'r:')
        # Now plot the ratios.
        ww = np.nonzero( pkd[:,0]<knl )[0]
        r0 = np.interp(pkd[ww,0],kk,pz0)/pk0[ww]
        r2 = np.interp(pkd[ww,0],kk,pz2)/pk2[ww]
        ax[1,ii].plot(pkd[ww,0],r0,'b:')
        ax[1,ii].plot(pkd[ww,0],r2,'r:')
        # Add a grey shaded region.
        ax[1,ii].fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
        ax[1,ii].fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
        # put on a line for knl.
        ax[0,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        ax[0,ii].text(0.9*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='right',va='center')
        ax[1,ii].plot([knl,knl],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0,ii].set_xlim(0.02,1.0)
        ax[0,ii].set_ylim(2.0,3e4)
        ax[0,ii].set_xscale('log')
        ax[0,ii].set_yscale('log')
        ax[0,ii].text(0.025,5.,"$z={:.1f}$".format(zz))
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
    ax[1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0,0].set_ylabel(r'$P_\ell(k)\quad [h^{-3}{\rm Mpc}^3]$')
    ax[1,0].set_ylabel(r'$P_Z/P_{N-body}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig('zeld_pks.pdf')
    #









if __name__=="__main__":
    make_pkr_plot()
    make_pks_plot()
    #
