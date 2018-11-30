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
ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
#ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'



def plotH1pk(fname, nc=256, fsize=15):
    '''plot the power spectrum of halos and H1 on 'nc' grid'''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    fig, ax = plt.subplots(1, 2, figsize = (9,4)) # 
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        print(zz)
        path = project + sim+ '/fastpm_%0.4f/'%aa
        k, pkm = np.loadtxt(path + 'pkm.txt').T[0], np.loadtxt(path + 'pkm.txt').T[1] 
        pkh = np.loadtxt(path + 'pkhmass.txt').T[1]
        pkhp = np.loadtxt(path + 'pkhpos.txt').T[1]
        pkh1 = np.loadtxt(path + 'pkh1mass.txt').T[1]
        pkhm = np.loadtxt(path + 'pkhmassxm.txt').T[1]
        pkh1m = np.loadtxt(path + 'pkh1massxm.txt').T[1]
        pkhpm = np.loadtxt(path + 'pkhposxm.txt').T[1]
        
        ax[0].plot(k, pkh, label=zz, color='C%d'%i)
        ax[0].plot(k, pkhp, label=zz, ls="--", color='C%d'%i)
        ax[1].plot(k, pkh1, label=zz, color='C%d'%i) # 
    
    ax[1].legend(ncol=2, fontsize=13)
    ax[0].set_ylabel('P(k) Halos (Mass/Position)', fontsize=fsize)
    ax[1].set_ylabel('P(k) H1 Mass', fontsize=fsize)

    for axis in ax: 
        axis.set_xlim(2e-2, 1)
        axis.set_ylim(9e1, 5e4)
        axis.loglog()
        #axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout()
    fig.savefig(figpath + fname)



def plotH1bias(fname, nc=256, fsize=15):
    '''plot the power spectrum of halos and H1 on 'nc' grid'''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    fig, ax = plt.subplots(1, 2, figsize = (9,4), sharex=True, sharey=True ) # 
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        print(zz)
        path = project + sim+ '/fastpm_%0.4f/'%aa
        k, pkm = np.loadtxt(path + 'pkm.txt').T[0], np.loadtxt(path + 'pkm.txt').T[1] 
        pkh = np.loadtxt(path + 'pkhmass.txt').T[1]
        pkhp = np.loadtxt(path + 'pkhpos.txt').T[1]
        pkh1 = np.loadtxt(path + 'pkh1mass.txt').T[1]
        pkhm = np.loadtxt(path + 'pkhmassxm.txt').T[1]
        pkh1m = np.loadtxt(path + 'pkh1massxm.txt').T[1]
        pkhpm = np.loadtxt(path + 'pkhposxm.txt').T[1]

        #b1h = pkh/pkd
        #b1hp = pkhp/pkd
        b1h1 = pkh1m/pkm
        b1h1sq = pkh1/pkm
        ax[0].plot(k, b1h1, label=zz, color='C%d'%i)
        ax[1].plot(k, b1h1sq**0.5, label=zz, color='C%d'%i)
    
    ax[1].legend(ncol=2, fontsize=13, loc='lower right')
    ax[0].set_ylabel('$P_{H1-m}/P_m$', fontsize=fsize)
    ax[1].set_ylabel('$\sqrt{P_{H1}/P_m}$', fontsize=fsize)

    for axis in ax: 
        axis.set_ylim(0, 10)
        axis.set_xscale('log')
        axis.set_xlabel('k(h/Mpc)')
        #axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout()
    fig.savefig(figpath + fname)



def plotH1mass(fname, fsize=15):
    '''plot cdf of H1'''
    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        print(zz)
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
        #axis.grid(which='both', lw=0.5, color='gray')
    fig.tight_layout()
    fig.savefig(figpath + fname)



def plot_params(fname, fsize=15):
    '''plot params from the Table 6 of arxiv:1804.09180'''

    zz = np.linspace(0, 6)
    zzp = np.arange(0, 6, 1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(zz, dohod.alphaz(zz))
    ax[0].plot(zz, dohod.alphaz(zz, mwfits=True), ls="--")
    ax[0].plot(zzp, dohod.alphaz(zzp), marker='o')
    ax[0].set_title('alpha(z)')

    ax[1].plot(zz, dohod.M0z(zz))
    ax[1].plot(zzp, dohod.M0z(zzp), marker='o')
    ax[1].plot(zz, dohod.M0z(zz, mwfits=True), ls="--")
    ax[1].set_yscale('log')
    ax[1].set_title('M0(z)')

    ax[2].plot(zz, dohod.Mminz(zz))
    ax[2].plot(zzp, dohod.Mminz(zzp), marker='o')
    ax[2].plot(zz, dohod.Mminz(zz, mwfits=True), ls="--")
    ax[2].set_yscale('log')
    ax[2].set_title('Mmin(z)')
    fig.suptitle('Parameters from Table 6 of arxiv:1804.09180')
    fig.savefig(figpath + fname)


if __name__=="__main__":

    plot_params('paramz.png')
    #plotH1mass(prefix + '-H1mass.png')
    plotH1pk(prefix + '-H1pk.png')
    plotH1bias(prefix + '-H1bias.png')
