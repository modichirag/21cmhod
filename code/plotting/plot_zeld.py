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
fsize = 11
fontmanage = font_manager.FontProperties(family='serif', style='normal',
    size=fsize, weight='normal', stretch='normal')
font = {'family': fontmanage.get_family()[0],
        'style':  fontmanage.get_style(),
        'weight': fontmanage.get_weight(),
        'size': fontmanage.get_size(),
        }

from nbodykit.cosmology.cosmology import Cosmology
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
h = cosmodef['h']
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


def thermalnoise(stage2=True, mK=False):
    fsky21 = 20000/41252;
    Ns = 256
    if not stage2: Ns = 32
    Ds = 6
    n0 = (Ns/Ds)**2
    Ls = Ds* Ns
    npol = 2
    S21 = 4 *np.pi* fsky21
    t0 = 5*365*24*60*60
    Aeff = np.pi* (Ds/2)**2;
    nu0 = 1420*1e6
    wavez = lambda z: 0.211 *(1 + z)
    chiz = lambda z : cosmo.comoving_distance(z)
    #defintions
    n = lambda D: n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * np.exp(-(D/Ls)**6.8390)
    Tb = lambda z: 180/(cosmo.efunc(z)) *(4 *10**-4 *(1 + z)**0.6) *(1 + z)**2*h 
    FOV= lambda z: (1.22* wavez(z)/Ds)**2; #why is Ds here
    Ts = lambda z: (50 + 2.7 + 25 *(1420/400/(1 + z))**-2.75) * 1000;
    u = lambda k, mu, z: k *np.sqrt(1 - mu**2)* chiz(z) /(2* np.pi)#* wavez(z)**2
    #terms
    d2V = lambda z: chiz(z)**2* 3* 10**5 *(1 + z)**2 /cosmo.efunc(z)/100 
    fac = lambda z: Ts(z)**2 * S21 / Aeff **2 * (wavez(z))**4 /FOV(z) 
    # fac = lambda z: Ts(z)**2 * S21 /Aeff **2 * ((1.22 * wavez(z))**2)**2 #/FOV(z)/1.22**4
    cfac = 1 /t0/ nu0 / npol
    #
    if mK:  Pn = lambda k, mu, z: cfac *fac(z) *d2V(z) / (n(u(k, mu, z)) * wavez(z)**2)
    else: Pn =  lambda k, mu, z: cfac *fac(z) *d2V(z) / (n(u(k, mu, z)) * wavez(z)**2) / Tb(z)**2
    return Pn



def make_pkr_plot():
    """Does the work of making the real-space P(k) figure."""
    zlist = [2.000,6.000]
    #    b1lst = [0.900,2.750]
    #    b2lst = [0.800,6.250]
    #    alpha = [1.500,0.145]
    #
    b1lst = [0.920,2.750]
    b2lst = [-.125,5.788]
    bnlst = [3.713,1.00]
    alpha = [1.500,0.150]
    #bnlst = [3.613,0.650]
    #alpha = [1.500,0.145]


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

##        pzh = (1+alpha[ii]*kk**2)*pkz[:,1]+b1*pkz[:,2]+b2*pkz[:,3]+\
##              b1**2*pkz[:,4]+b2**2*pkz[:,5]+b1*b2*pkz[:,6]
##        pzx = (1+alpha[ii]*kk**2)*pkz[:,1]+0.5*b1*pkz[:,2]+0.5*b2*pkz[:,3]
##        pzm = (1+alpha[ii]*kk**2)*pkz[:,1]
        
        alh = alpha[ii] + bnlst[ii]
        alx = alpha[ii] + 0.5*bnlst[ii]
        alm = alpha[ii]
        pzh = (1+alh*kk**2)*pkz[:,1]+b1*pkz[:,2]+b2*pkz[:,3]+\
              b1**2*pkz[:,4]+b2**2*pkz[:,5]+b1*b2*pkz[:,6]
        pzx = (1+alx*kk**2)*pkz[:,1]+0.5*b1*pkz[:,2]+0.5*b2*pkz[:,3]
        pzm = (1+alm*kk**2)*pkz[:,1]

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
        ax[0,ii].plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
        ax[1,ii].plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
        ax[0,ii].axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        ax[1,ii].axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        if ii == 0:ax[0,ii].text(1.1*knl,1e4,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='left',va='center', fontdict=font)

        #cosmic variance
        ff = open(db + 'HI_pks_1d_{:06.4f}.txt'.format(aa))
        tmp = ff.readline()
        sn = float(tmp.split('=')[1].split('.\n')[0])
        print('shotnoise = ', sn)
        kk = pkd[:, 0]#np.logspace(-2, 0, 1000)
        Nk = 4*np.pi*kk[:-1]**2*np.diff(kk)*bs**3 / (2*np.pi)**3
        std = (2/Nk *(pkh[:-1]+sn)**2)**0.5/pkh[:-1]
        #std = np.sqrt(2/Nk)
        #print(Nk)
        ax[1,ii].fill_between(kk[:-1], 1-std, 1+std, 
                              color='magenta',alpha=0.1)

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
    #a2lst = [3.800,2.000]
    a2lst = [5.00,2.500]

    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(6,4),sharex=True,sharey='row',\
                          gridspec_kw={'height_ratios':[3,1,1]})
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
        pk4 = pkd[:,3]
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
        ax[2,ii].plot(pkd[ww,0],1/r2,'C1', lw=1.2)

        #cosmic variance
        ff = open(db + 'HI_pks_1d_{:06.4f}.txt'.format(aa))
        tmp = ff.readline()
        sn = float(tmp.split('=')[1].split('.\n')[0])
        print(zz, 'shotnoise = ', sn)
        kk = pkd[:, 0]#np.logspace(-2, 0, 1000)
        Nk = 4*np.pi*kk[:-1]**2*np.diff(kk)*bs**3 / (2*np.pi)**3
        
        #std2 = (2/Nk *((pk0[:-1]+sn)**2 + pk2[:-1]**2/5))**0.5 / pk0[:-1]
        #std3 = (2/Nk)**0.5
        muu = np.linspace(-1, 1, 1000).reshape(1, -1)
        l0 = 1 + muu*0
        l2 = 1*(3*muu**2-1)/2.
        l4 = (35*muu**4-30*muu**2+3)/8
        integrand = (l0*pk0[:-1].reshape(-1, 1) + l2*pk2[:-1].reshape(-1, 1) + l4*pk4[:-1].reshape(-1, 1) 
                     + sn)**2 * l0**2
        std20 = ( 2/Nk *(np.trapz(integrand, muu) * (2*0+1)**2/2) )**0.5 / pk0[:-1]
        #std20 = ( 1/kk[:-1]**2 *(np.trapz(integrand, muu) * (2*0+1)**2/2) )**0.5 
        ax[1,ii].fill_between(kk[:-1], 1-std20, 1+std20, color='C2',alpha=0.5)
        integrand = (l0*pk0[:-1].reshape(-1, 1) + l2*pk2[:-1].reshape(-1, 1) + l4*pk4[:-1].reshape(-1, 1) 
                     + sn)**2 * l2**2
        std22 = ( 2/Nk *(np.trapz(integrand, muu) * (2*2+1)**2/2) )**0.5 / pk2[:-1]
        ax[2,ii].fill_between(kk[:-1], 1-std22, 1+std22, color='C2',alpha=0.5)

        #themal noise
        Nk *= 0.6 #wedge cut

        cip = ['r', 'b']
        for ip, Pn in enumerate([thermalnoise(stage2=True), thermalnoise(stage2=False)]):
        #for ip, Pn in enumerate([thermalnoise(stage2=True)]):
            if ii ==1 and ip == 1: continue
            pthmu = Pn(kk[:-1].reshape(-1, 1), muu, zz)
            pthsq = np.trapz(pthmu**2, muu) 
            pl0 = np.trapz(pthmu*l0, muu)*1/2. 
            pl2 = np.trapz(pthmu*l2, muu)*5/2. 

            integrand = (l0*pk0[:-1].reshape(-1, 1) + l2*pk2[:-1].reshape(-1, 1) + l4*pk4[:-1].reshape(-1, 1) 
                         + sn + pthmu)**2 * l0**2
            std0 = ( 2/Nk *(np.trapz(integrand, muu) * (2*0+1)**2/2) )**0.5 / pk0[:-1]
            #print((std0/std20)[kk[:-1]<1])
            #std4 = (2/Nk *((pk0[:-1]+sn)**2 + pk2[:-1]**2/5 + pthsq + 2*(pk0[:-1]+sn)*pl0 + 2*pk2[:-1]*pl2/5))**0.5 / pk0[:-1]
            ax[1,ii].plot(kk[:-1], 1-std0,  color=cip[ip],alpha=0.7, lw=0.7, ls="--")
            ax[1,ii].plot(kk[:-1], 1+std0,  color=cip[ip],alpha=0.7, lw=0.7, ls="--")

            integrand = (l0*pk0[:-1].reshape(-1, 1) + l2*pk2[:-1].reshape(-1, 1) + l4*pk4[:-1].reshape(-1, 1) 
                         + sn + pthmu)**2 * l2**2
            std2 = ( 2/Nk *(np.trapz(integrand, muu) * (2*2+1)**2/2) )**0.5 / pk2[:-1]
            ax[2,ii].plot(kk[:-1], 1-std2,  color=cip[ip],alpha=0.7, lw=0.7, ls="--")
            ax[2,ii].plot(kk[:-1], 1+std2,  color=cip[ip],alpha=0.7, lw=0.7, ls="--")
        

        # put on a line for knl.
        # Add a grey shaded region.
        for axis in ax[1:, ii]:
            axis.fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
            axis.fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
        for axis in ax[:,ii].flatten():
            axis.plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
            axis.axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        #ax[0,ii].plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
        #ax[1,ii].axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        #text
        if ii==0: ax[0,ii].text(1.1*knl,25,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='left',va='center', fontdict=font)
        ax[0,ii].text(0.025,25.,"$z={:.1f}$".format(zz), fontdict=font)
        ax[1,0].text(0.025,0.92,"$\ell=0$", fontdict=font)
        ax[2,0].text(0.025,0.92,"$\ell=2$", fontdict=font)


##    tmp = np.loadtxt('/global/project/projectdirs/m3127/H1mass/highres/2560-9100-fixed/fastpm_0.3333/pkm.txt').T
##    kk = tmp[0]#np.logspace(-2, 0, 1000)
##    Nk = 4*np.pi*kk[:-1]**2*np.diff(kk)*256**3 /  (2*np.pi)**3
##    print(tmp[2, :-1]**-1*Nk)
##
    # Tidy up the plot.
    ax[0,0].set_ylim(10.0,7e4)
    ax[0,0].set_yscale('log')
    ax[1,0].set_ylim(0.90,1.1)
    ax[2,0].set_ylim(0.90,1.1)
    for axis in ax.flatten():
        axis.set_xscale('log')
        axis.set_xlim(0.02,1.)
            
        #
    # Suppress the y-axis labels on not-column-0.
    for ii in range(1,ax.shape[1]):
        ax[0,ii].get_yaxis().set_visible(False)
        ax[1,ii].get_yaxis().set_visible(False)
    # Put on some more labels.
    ax[0,1].legend(prop=fontmanage, ncol=2, loc=1)

    ax[-1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[-1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$P_\ell(k)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
    #ax[1,0].set_ylabel(r'$P_{\ell=0, N-body}/P_{\ell=0, Z}$', fontdict=font)
    ax[1,0].set_ylabel(r'$P_N/P_Z$', fontdict=font)
    ax[2,0].set_ylabel(r'$P_N/P_Z$', fontdict=font)

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
    #make_pkr_plot()
    make_pks_plot()
    #
