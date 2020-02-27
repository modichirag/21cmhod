#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
from nbodykit.cosmology.cosmology import Cosmology
from scipy.optimize import minimize
#
from matplotlib import rc, rcParams, font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
rcParams['font.family'] = 'serif'
fsize = 11
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
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
args = parser.parse_args()
if args.model == None:
    import sys
    print('Specify a model name')
    sys.exit()
print(args, args.model)

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
h = cosmodef['h']


model = args.model #'ModelD'
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
bs = 256
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
dpath = '../../data/outputs/%s/%s/'%(suff, model)
tb = '../..//theory/'
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass

svfilter = True
if bs == 256: winsize = 7
elif bs == 1024: winsize  = 39
polyorder = 3


####################################################
def volz(z, dz=0.1, sky=20000):
    chiz = cosmo.comoving_distance(z)
    dchiz = cosmo.comoving_distance(z+dz)-cosmo.comoving_distance(z-dz)
    fsky = sky/41252;
    vol = 4*np.pi*chiz**2 * dchiz * fsky
    return vol

def wedge(z, D=6 *0.7**0.5, att='opt'):
    chiz = cosmo.comoving_distance(z)
    hz = cosmo.efunc(z)*100
    R = chiz*hz/(1+z)/2.99e5
    if att == 'opt': thetafov = 1.22 * 0.211 * (1+z)/D/2
    elif att == 'pess': thetafov = 1.22 * 0.211 * (1+z)/D/2 *3
    elif att == 'nope': thetafov = np.pi/2.
    X = np.sin(thetafov) * R
    mu = X/(1+X**2)**0.5
    return mu


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
    Aeff = np.pi* (Ds/2)**2 *0.7 #effective
    nu0 = 1420*1e6
    wavez = lambda z: 0.211 *(1 + z)
    chiz = lambda z : cosmo.comoving_distance(z)
    #defintions
    n = lambda D: n0 *(0.4847 - 0.33 *(D/Ls))/(1 + 1.3157 *(D/Ls)**1.5974) * np.exp(-(D/Ls)**6.8390)
    Tb = lambda z: 180/(cosmo.efunc(z)) *(4 *10**-4 *(1 + z)**0.6) *(1 + z)**2*h 
    FOV= lambda z: (1.22* wavez(z)/Ds)**2; #why is Ds here
    Ts = lambda z: (55 + 30 + 2.7 + 25 *(1420/400/(1 + z))**-2.75) * 1000;
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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


####################################################


def make_bao_plot(fname):
    """Does the work of making the BAO figure."""

    winsize = 11
    polyorder = 3
    svfilter = True

    zlist = [2.0, 6.0]
    clist = ['b','c','g','m','r','y']
    Nmus = [int(8), int(20)]


    fig,ax = plt.subplots(1,2,figsize=(9,4),sharey=True)
    ii,jj  = 0,0

    for iz, zz in enumerate(zlist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        Nmu = Nmus[iz]
        mue = np.linspace(0, 1, Nmu+1)
        mus = 0.5*(mue[1:] + mue[:-1])


        mumin = wedge(zz, att='pess')
        ind = np.where(mus > mumin)[0][0]-1
        if iz == 0: ind +=1
        print(ind, mus[ind], mumin, Nmu-ind)
        inds = np.arange(ind, Nmu, 1)
        inds = [inds[:inds.size//2], inds[inds.size//2:]]
        mu1, mu2 = [mus[inds[0][0]], mus[inds[0][-1]]],  [mus[inds[1][0]], mus[inds[1][-1]]]
        mu1e, mu2e = [mue[inds[0][0]], mue[1+inds[0][-1]]],  [mue[inds[1][0]], mue[1+inds[1][-1]]]
        print(mu1, mu2)

        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        #redshift space
        pks = np.loadtxt(dpath + "HI_pks_mu_{:02d}_{:06.4f}.txt".format(Nmu, aa))[5:,:]
        print(np.isnan(pks).sum())
        nans, x= nan_helper(pks[:, 1:])
        pks[:, 1:][nans]= np.interp(x(nans), x(~nans), pks[:, 1:][~nans])
        pks = (pks[1::2] + pks[::2])*0.5 #Smooth
        print(np.isnan(pks).sum())
        
        pk1, pk2 = pks[:, inds[0]].sum(axis=1)/len(inds[0]), pks[:, inds[1]].sum(axis=1)/len(inds[1])
        kk = pks[:, 0]
        pks = [pk1, pk2]

        pkb = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        bb = pkb[1:6,1].mean()
        ff = (0.31/(0.31+0.69*aa**3))**0.55
        klin, plin = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa), unpack=True)
        pklins = []
        for imu, mu in enumerate([mu1, mu2]):
            mumin,mumax = mu
            mux = np.linspace(mumin, mumax, 100)
            kf = (bb+ff*mux**2)**2
            pklins.append(plin*np.trapz(kf, mux)/(mumax-mumin))
        pklin1, pklin2 = pklins
        
        # Take out the broad band.
        if not svfilter: # Use smoothing spline as broad-band/no-wiggle.
            knots=np.arange(0.05,0.5,0.03)
            ss1  = Spline(kk, pk1 ,t=knots)
            rat1 = pk1/ss1(kk)
            ss2  = Spline(kk, pk2 ,t=knots)
            rat2 = pk2/ss2(kk)
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ss1  = savgol_filter(pk1, winsize,polyorder=polyorder)
            rat1 = pk1/ss1 
            ss2  = savgol_filter(pk2, winsize,polyorder=polyorder)
            rat2 = pk2/ss2 
#            ss1lin  = savgol_filter(np.interp(kk, klin, pklin1), winsize,polyorder=polyorder)
#            rat1lin = np.interp(kk, klin, pklin1)/ss1lin
#            ss2lin  = savgol_filter(np.interp(kk, klin, pklin2), winsize,polyorder=polyorder)
#            rat2lin = np.interp(kk, klin, pklin2)/ss2lin
            ss1lin  = savgol_filter(pklin1, 147 ,polyorder=polyorder)
            rat1lin = pklin1/ss1lin
            ss2lin  = savgol_filter(pklin2, 147 ,polyorder=polyorder)
            rat2lin = pklin2/ss2lin

        ax[iz].axhline(1, color='k', lw=0.5)
        ax[iz].axhline(1.2, color='k', lw=0.5)
        ax[iz].text(0.3,1.25,"$z={:.1f}$".format(zz), fontdict=font)

        #cosmic variance
        ff = open(dpath + 'HI_pks_1d_{:06.4f}.txt'.format(aa))
        tmp = ff.readline()
        sn = float(tmp.split('=')[1].split('.\n')[0])
        print(zz, 'shotnoise = ', sn)
        Nk = 4*np.pi*kk**2*np.diff(kk)[0]*bs**3 / (2*np.pi)**3
        Pn, Pn2 = thermalnoise(stage2=False), thermalnoise(stage2=True)
        cip = ['r', 'g']
        rats = [rat1, rat2]
        ratslin = [rat1lin, rat2lin]
        sss = [ss1, ss2]
        stages =  ['HIRAX', 'Stage II']
        for ip, pn in enumerate([Pn, Pn2]):
            sky = [15000, 20000][ip]
            V = volz(zz, sky=sky)
            for imu, mu in enumerate([mu1e, mu2e]):
                mumin, mumax = mu
                print(mumin, mumax)
                mux = np.linspace(mumin, mumax).reshape(1,-1)
                Nkmu = Nk/1*(mumax-mumin) *V/bs**3
                #std = ( 2/Nkmu * (sn+pks[imu])**2)**0.5 /pks[imu]
                std = ( 2/Nkmu * (sn+pks[imu]+np.trapz(pn(kk.reshape(-1,1), mux, zz), mux)/ (mumax-mumin))**2)**0.5 / sss[imu]#pks[imu]
                yerr = std 
                lbl = None
                if ip == 1: lbl = r'$\mu = (%.2f, %.2f)$'%(mumin, mumax)
                if iz == 1 and ip == 0: 
                    #only for labeling
                    ax[iz].errorbar([0,0.01], [1, 2], .1, color='k', \
                                                        ecolor='%s'%cip[imu], label=stages[imu], elinewidth=3, alpha=0.5, lw=0.)
                else: 
                    ax[iz].errorbar(kk, rats[imu]+0.2*imu, yerr, ecolor='%s'%cip[ip], color='C%d'%imu, label=lbl, \
                                      elinewidth=3, alpha=0.5, lw=2)
                    if iz == 1:
                        #to make same thickness
                        ax[iz].plot(kk, rats[imu]+0.2*imu, color='C%d'%imu, alpha=0.5, lw=2)
                #Make linear
                if iz == 0 and ip == 0 and imu == 0: 
                    lbllin = 'Linear'
                else : lbllin = None
                ax[iz].plot(klin, ratslin[imu]+0.2*imu, 'k--', lw=1, label=lbllin)
                        

    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5,prop=fontmanage)
        ax[ii].set_xlim(0.045,0.4)
        ax[ii].set_ylim(0.83,1.32)
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0].set_ylabel(r'$P(k)/P_{\rm nw}(k)$+offset', fontdict=font)
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)


    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #





  



def make_mu_plot(fname):
    """Does the work of making the BAO figure."""
    zlist = [2.0, 6.0]
    clist = ['b','c','g','m','r','y']
    Nmus = [8, 20]
    
    ii,jj  = 0,0
    zlist = [2.000,6.000]
    #zlist = [2.000]
    #Numbers for first try
##    b1lst = [0.826, 2.62]
##    b2lst = [0.55, 7.375]
##    a0lst = [1.63, -1.30]
##    a2lst = [-1.73,0.500]
##    a4lst = [0.02,-1.000]
#Nubers for second try
    b1lst = [0.79, 2.62]
    b2lst = [1.60, 7.375]
    a0lst = [-3.33, -1.30]
    a2lst = [-3.03,0.500]
    a4lst = [-40.2,-1.000]

    # Now make the figure.
    fig,ax = plt.subplots(3,2,figsize=(6,4),sharex=True,sharey='row',\
                          gridspec_kw={'height_ratios':[3,1,1]})
    for ii in range(len(zlist)):
        Nmu = Nmus[ii]
        mue = np.linspace(0, 1, Nmu+1)
        mus = 0.5*(mue[1:] + mue[:-1])

        zz = zlist[ii]
        aa = 1.0/(1.0+zz)
        bb = b1lst[ii] + 1.0
        b1 = b1lst[ii]
        b2 = b2lst[ii]
        a0 = a0lst[ii]
        a2 = a2lst[ii]
        a4 = a4lst[ii]
        ak = [a0, a2, a4]

        mumin = wedge(zz, att='pess')
        ind = np.where(mus > mumin)[0][0]-1
        if ii == 0: ind +=1
        print(ind, mus[ind], mumin, Nmu-ind)
        inds = np.arange(ind, Nmu, 1)
        inds = [inds[:inds.size//2], inds[inds.size//2:]]
        mu1, mu2 = [mus[inds[0][0]], mus[inds[0][-1]]],  [mus[inds[1][0]], mus[inds[1][-1]]]
        mu1e, mu2e = [mue[inds[0][0]], mue[1+inds[0][-1]]],  [mue[inds[1][0]], mue[1+inds[1][-1]]]
        print(mu1, mu2)

        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        #redshift space
        pks = np.loadtxt(dpath + "HI_pks_mu_{:02d}_{:06.4f}.txt".format(Nmu, aa))[5:,:]
        print(np.isnan(pks).sum())
        nans, x= nan_helper(pks[:, 1:])
        pks[:, 1:][nans]= np.interp(x(nans), x(~nans), pks[:, 1:][~nans])
        #pks = (pks[1::2] + pks[::2])*0.5 #Smooth
        print(np.isnan(pks).sum())
        
        pk1, pk2 = pks[:, inds[0]].sum(axis=1)/len(inds[0]), pks[:, inds[1]].sum(axis=1)/len(inds[1])
        kk = pks[:, 0]
        pks = [pk1, pk2]
        ax[0,ii].plot(kk,pk1,'C0-',alpha=0.75, lw=1.2, label=r'$\mu = (%.2f, %.2f)$'%(mu1e[0], mu1e[1]))
        ax[0,ii].plot(kk,pk2,'C1-',alpha=0.75, lw=1.2, label=r'$\mu = (%.2f, %.2f)$'%(mu2e[0], mu2e[1]))

        # Now Zeldovich.
        pz0 = np.loadtxt(tb+"zeld_{:6.4f}.pk0".format(aa))
        pz2 = np.loadtxt(tb+"zeld_{:6.4f}.pk2".format(aa))
        pz4 = np.loadtxt(tb+"zeld_{:6.4f}.pk4".format(aa))
        kz  = pz0[:,0]
        pk  = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        knl = 1.0/np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        plf = [pz0, pz2, pz4]

        def getpkmu(mubin, pls):
            mumin, mumax = mubin 
#            mux = np.linspace(mumin, mumax, 1000)
#            l0 = np.trapz(1 + mux*0, mux)
#            l2 = np.trapz(1*(3*mux**2-1)/2., mux)
#            l4 = np.trapz((35*mux**4-30*mux**2+3)/8, mux)
#            pkmu = (pls[0]*l0 + pls[1]*l2 + pls[2]*l4) / (mumax-mumin)
#
            mux = np.linspace(mumin, mumax, 1000).reshape(1, -1)
            l0 = 1 + mux*0
            l2 = 1*(3*mux**2-1)/2.
            l4 = (35*mux**4-30*mux**2+3)/8
            pkmu = (pls[0].reshape(-1, 1)*l0 + pls[1].reshape(-1, 1)*l2 + pls[2].reshape(-1, 1)*l4)
            pkmu = np.trapz(pkmu, mux) / (mumax-mumin)
            return pkmu

        def getzeldp(p):
            b11, b2, a0, a2, a4 = p
            ak = [a0, a2, a4]
            def getzeld(pk, ia):
                return  (1+ 1*ak[ia]*kz**2)*pk[:,1]+b1*pk[:,2]+b2*pk[:,3]+ \
                    b1**2*pk[:,4]+b2**2*pk[:,5]+b1*b2*pk[:,6]
            pls = getzeld(pz0, 0), getzeld(pz2, 1), getzeld(pz4, 2)
            pk1z, pk2z = getpkmu(mu1, pls), getpkmu(mu2, pls)
            pk1z, pk2z = np.interp(kk, kz, pk1z), np.interp(kk, kz, pk2z)
            return pk1z, pk2z

        def fitps(p, wts=1, dofit=True):
            b1, b2, a0, a2, a4 = p
            ak = [a0, a2, a4]
            if b2 < -2:  wts = 1e5
            def getzeld(pk, ia):
                return  (1+ 1* ak[ia]*kz**2)*pk[:,1]+b1*pk[:,2]+b2*pk[:,3]+ \
                    b1**2*pk[:,4]+b2**2*pk[:,5]+b1*b2*pk[:,6]

            pls = getzeld(pz0, 0), getzeld(pz2, 1), getzeld(pz4, 2)
            pk1z, pk2z = getpkmu(mu1, pls), getpkmu(mu2, pls)
            pk1z, pk2z = np.interp(kk, kz, pk1z), np.interp(kk, kz, pk2z)
            if dofit: return (((1-pk1z/pk1)**2 + (1-pk2z/pk2)**2 )* wts).sum()**0.5

            else: return pk1z, pk2z

        #
        #
        ww = np.nonzero( kk[:]<knl )[0]
        test = [b1,b2, a0, a2, a4]
        pk1z, pk2z = getzeldp(test)
        ax[0,ii].plot(kk, pk1z,'C0--',alpha=0.75, lw=1.2)
        ax[0,ii].plot(kk, pk2z,'C1--',alpha=0.75, lw=1.2)
        r0 = pk1z[ww]/pk1[ww]
        r2 = pk2z[ww]/pk2[ww]
        ax[1,ii].plot(kk[ww],1/r0,'C0', lw=1.2)
        ax[2,ii].plot(kk[ww],1/r2,'C1', lw=1.2)

##        if ii == 0:
##            wts = np.ones_like(kk)
##            wts[kk>0.4] = 0
##            wts[kk<0.08] = 0
##            tominimize = lambda p : fitps(p, wts=wts)
##            init = [1, 1, 0, 0, 0]
##            fit = minimize(tominimize, init, method='Nelder-Mead',options={'maxiter':5000}).x
##            print(init, fit)
##            pk1z, pk2z = fitps(fit, dofit=False)
##            ax[0,ii].plot(kk, pk1z,'C0',alpha=0.75, lw=1.2)
##            ax[0,ii].plot(kk, pk2z,'C1',alpha=0.75, lw=1.2)
##            r0 = pk1z[ww]/pk1[ww]
##            r2 = pk2z[ww]/pk2[ww]
##            ax[1,ii].plot(kk[ww],1/r0,'C2--', lw=1.2)
##            ax[2,ii].plot(kk[ww],1/r2,'C3--', lw=1.2)
##

        #cosmic variance
        ff = open(dpath + 'HI_pks_1d_{:06.4f}.txt'.format(aa))
        tmp = ff.readline()
        sn = float(tmp.split('=')[1].split('.\n')[0])
        print(zz, 'shotnoise = ', sn)
        Nk = 4*np.pi*kk**2*np.diff(kk)[0]*bs**3 / (2*np.pi)**3
        Pn, Pn2 = thermalnoise(stage2=False), thermalnoise(stage2=True)
        cip = ['r', 'k']
        for ip, pn in enumerate([Pn, Pn2]):
            for imu, mu in enumerate([mu1, mu2]):
                if ii == 1 and ip == 0: continue
                mumin, mumax = mu
                mux = np.linspace(mumin, mumax).reshape(1,-1)
                #Nkmu = Nk/2/np.pi*(mumax-mumin)
                Nkmu = Nk/1*(mumax-mumin)
                std = ( 2/Nkmu * (sn+pks[imu])**2)**0.5 /pks[imu]
                ax[imu+1, ii].fill_between(kk, 1-std, 1+std, color='g', alpha=0.1)
                std = ( 2/Nkmu * (sn+pks[imu]+np.trapz(pn(kk.reshape(-1,1), mux, zz), mux)/ (mumax-mumin))**2)**0.5 /pks[imu]
                #ax[imu+1, ii].fill_between(kk, 1-std, 1+std, color=cip[ip], alpha=0.2)
                ax[imu+1, ii].plot(kk, 1-std, '--', color=cip[ip], lw=1, alpha=0.7)
                ax[imu+1, ii].plot(kk, 1+std, '--', color=cip[ip], lw=1, alpha=0.7)
                #ax[imu+1, ii].plot(kk, 1+std, 'C%d--'%ip)
                
        

##
        # put on a line for knl.
        # Add a grey shaded region.
        for axis in ax[1:, ii].flatten():
            axis.fill_between([1e-5,3],[0.95,0.95],[1.05,1.05],\
                   color='lightgrey',alpha=0.25)
            axis.fill_between([1e-5,3],[0.98,0.98],[1.02,1.02],\
                   color='darkgrey',alpha=0.5)
            axis.set_ylim(0.9, 1.1)
        for axis in ax[:,ii].flatten():
            axis.plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
            axis.axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        #ax[0,ii].plot([knl,knl],[1e-10,1e10],'-.',color='k', alpha=0.2)
        #ax[1,ii].axvline(0.75*knl,ls=':',color='k',alpha=0.2)
        #text
        if ii==0: ax[0,ii].text(1.1*knl,25,r'$k_{\rm nl}$',color='darkgrey',\
                       ha='left',va='center', fontdict=font)
        ax[0,ii].text(0.025,500.,"$z={:.1f}$".format(zz), fontdict=font)
        #ax[1,0].text(0.025,0.92,"$\ell=0$", fontdict=font)
        #ax[2,0].text(0.025,0.92,"$\ell=2$", fontdict=font)


    # Suppress the y-axis labels on not-column-0.
    for ii in range(1,ax.shape[1]):
        ax[0,ii].get_yaxis().set_visible(False)
        ax[1,ii].get_yaxis().set_visible(False)
    # Put on some more labels.

    for axis in ax[0, :]:
        axis.set_ylim(10, 4e4)
        axis.set_yscale('log')
        axis.set_xlim(2e-2, 1)
        axis.set_xscale('log')
        axis.legend(prop=fontmanage, ncol=1, loc=0)

    ax[-1,0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[-1,1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[0,0].set_ylabel(r'$P(k, \mu)\quad [h^{-3}{\rm Mpc}^3]$', fontdict=font)
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
    plt.savefig(fname)
    #
##
##
##
if __name__=="__main__":
    make_mu_plot(figpath + 'HI_noise_pkmu_%s.pdf'%model)
    make_bao_plot(figpath + 'HI_noise_bao_%s.pdf'%model)
    #
