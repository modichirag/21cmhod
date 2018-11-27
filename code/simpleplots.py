import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import sys
sys.path.append('./utils')
import tools, dohod             # 
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
figpath = './figs/'

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
#ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'



def plotH1pk(fname, nc=256, fsize=15):
    '''plot the power spectrum of halos and H1 on 'nc' grid'''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    fig, ax = plt.subplots(1, 2, figsize = (9,4)) # 
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        halos = BigFileCatalog(project + sim+ '/fastpm_%0.4f/halocat/'%aa)
        hmass, h1mass = halos["Mass"].compute(), halos['H1mass'].compute()
        hpos = halos['Position'].compute()
        

        hpmesh = pm.paint(hpos)
        hmesh = pm.paint(hpos, mass=hmass)
        h1mesh = pm.paint(hpos, mass=h1mass)
        pkh = FFTPower(hmesh/hmesh.cmean(), mode='1d').power
        pkhp = FFTPower(hpmesh/hpmesh.cmean(), mode='1d').power
        pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power
        
        ax[0].plot(pkh['k'], pkh['power'], label=zz, color='C%d'%i)
        ax[0].plot(pkh['k'], pkhp['power'], label=zz, ls="--", color='C%d'%i)
        ax[1].plot(pkh1['k'], pkh1['power'], label=zz, color='C%d'%i) # 
    
    ax[1].legend(ncol=2, fontsize=13)
    ax[0].set_ylabel('P(k) Halos (Mass/Position)', fontsize=fsize)
    ax[1].set_ylabel('P(k) H1 Mass', fontsize=fsize)

    for axis in ax: 
        axis.set_xlim(2e-2, 1)
        axis.set_ylim(9e1, 2e4)
        axis.loglog()
        axis.grid(which='both', lw=0.5, color='gray')
    fig.savefig(figpath + fname)


##
##def plotH1bias(fname, nc=256, fsize=15):
##    '''plot the power spectrum of halos and H1 on 'nc' grid'''
##
##    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
##
##    fig, ax = plt.subplots(1, 2, figsize = (9,4)) # 
##    for i, aa in enumerate(aafiles):
##        zz = zzfiles[i]
##        dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
##        halos = BigFileCatalog(project + sim+ '/fastpm_%0.4f/halocat/'%aa)
##        hmass, h1mass = halos["Mass"].compute(), halos['H1mass'].compute()
##        hpos = halos['Position'].compute()
##        
##        dmesh = pm.paint(dmcat['Position'])
##        hpmesh = pm.paint(hpos)
##        hmesh = pm.paint(hpos, mass=hmass)
##        h1mesh = pm.paint(hpos, mass=h1mass)
##        pkd = FFTPower(dmesh/dmesh.cmean(), mode='1d').power
##        pkh = FFTPower(hmesh/hmesh.cmean(), mode='1d').power
##        pkhp = FFTPower(hpmesh/hpmesh.cmean(), mode='1d').power
##        pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power
##        pkx = FFTPower(hmesh/hmesh.cmean(), dmesh/dmesh.cmean(), mode='1d').power
##        pkx1 = FFTPower(h1mesh/h1mesh.cmean(), dmesh/dmesh.cmean(), mode='1d').power
##
##        b1h = pkh['power']/pkd['power']
##        b1h1 = pkh1['power']/pkd['power']
##        b1hp = pkhp['power']/pkd['power']
##        k = pkh['k']
##        ax[0, 0].plot(k, b1h, label=zz, color='C%d'%i)
##        ax[0, 1].plot(k, b1h1, label=zz, color='C%d'%i)
##        ax[1, 0].plot(k, b1hp, label=zz, color='C%d'%i)
##    
##    ax[1].legend(fontsize=fsize)
##    ax[0].set_ylabel('P(k) Halos (Mass/Position)', fontsize=fsize)
##    ax[1].set_ylabel('P(k) H1 Mass', fontsize=fsize)
##
##    for axis in ax: 
##        axis.loglog()
##        axis.grid(which='both', lw=0.5, color='gray')
##    fig.savefig(figpath + fname)
##

def plotH1mass(fname, fsize=15):
    '''plot cdf of H1'''
    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        halos = BigFileCatalog(project + sim+ '/fastpm_%0.4f/halocat/'%aa)
        hmass, h1mass = halos["Mass"].compute(), halos['H1mass'].compute()
        ax[0].plot(hmass, h1mass, label=zz, lw=2)
        #     plt.plot(hmass, np.cumsum(h1mass)/h1mass.sum(), label=zz, lw=2)
        ax[1].plot(hmass[::-1], np.cumsum(h1mass[::-1])/h1mass.sum(), label=zz, lw=2)

    ax[0].loglog()
    ax[1].legend(ncol=2, fontsize=13)
    ax[1].set_xscale('log')
    ax[0].set_ylabel('$M_{H1}$', fontsize=fsize)
    ax[1].set_ylabel('$M_{H1}$ CDF', fontsize=fsize)
    for axis in ax: 
        axis.set_xlabel('$M_h$', fontsize=fsize)
        axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout()
    fig.savefig(figpath + fname)



def plot_params(fname, fsize=15):
    '''plot params from the Table 6 of arxiv:1804.09180'''

    zz = np.linspace(0, 6)
    zzp = np.arange(0, 6, 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(zz, dohod.alphaz(zz))
    ax[0].plot(zzp, dohod.alphaz(zzp), marker='o')
    ax[0].set_title('alpha(z)')

    ax[1].plot(zz, dohod.M0z(zz))
    ax[1].plot(zzp, dohod.M0z(zzp), marker='o')
    ax[1].set_yscale('log')
    ax[1].set_title('M0(z)')

    ax[2].plot(zz, dohod.Mminz(zz))
    ax[2].plot(zzp, dohod.Mminz(zzp), marker='o')
    ax[2].set_yscale('log')
    ax[2].set_title('Mmin(z)')
    fig.suptitle('Parameters from Table 6 of arxiv:1804.09180')
    fig.savefig(figpath + fname)


if __name__=="__main__":

    #plot_params('paramz.png')
    #plotH1mass(prefix + '-H1mass.png')
    plotH1pk(prefix + '-H1pk.png')
