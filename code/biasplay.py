import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
from nbodykit.source.mesh.field import FieldMesh
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
#
import sys
sys.path.append('./utils')
import tools, dohod             # 
from time import time
#
#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[4:]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 512
#ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'



def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    print('Mass of particle is = %0.2e'%mp)
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat



#def HI_mass(mhalo,aa):
#    """Makes a 21cm "mass" from a box of halo masses."""
#    print('Assigning weights')
#    zp1 = 1.0/aa
#    zz  = zp1-1
#    # Set the parameters of the HOD, using the "simple" form.
#    #   MHI ~ M0  x^alpha Exp[-1/x]       x=Mh/Mmin
#    # from the Appendix of https://arxiv.org/pdf/1804.09180.pdf, Table 6.
#    # Fits valid for 1<z<6:
#    mcut= 1e10*(6.11-1.99*zp1+0.165*zp1**2)
#    alp = (1+2*zz)/(2+2*zz)
#    # Work out the HI mass/weight per halo -- ignore prefactor.
#    xx  = mhalo/mcut+1e-10
#    mHI = xx**alp * np.exp(-1/xx)
#    # Scale to some standard number in the right ball-park.
#    #mHI*= 1.4e9*np.exp(-1.9*zp1+0.1*zp1**2)
#    # Scale to some standard number in the right ball-park. Second mail
#    mHI*= 2e9*np.exp(-1.9*zp1+0.07*zp1**2)
#    # Return the HI masses.
#    return(mHI)
#    #
#

def saveH1cat(aa, h1mass, halocat, fname, ofolder=None):
    '''save catalog...'''
    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)

    halocat = readincatalog(aa)
    #Do hod
    if ofolder is None:
        ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    halocat['H1mass'] = h1mass

    if save:
        colsave = [cols for cols in halocat.columns]
        colsave = ['ID', 'Position', 'Mass', 'H1mass']
        print(colsave)
        halocat.save(ofolder+fname, colsave)
        print('Halos saved at path\n%s'%ofolder)


def fiddlebias(aa, savecat=False, saveb=False, catname='h1cat', bname='h1bias.txt', ofolder=None):
    
    print('Read in catalogs')
    halocat = readincatalog(aa, matter=False)
    hpos = halocat['Position']
    hmass = halocat['Mass']
    h1mass = dohod.HI_mass(hmass, aa)
    dm = BigFileMesh(project + sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc, '1').paint()
    if ofolder is None:
        ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    
    if savecat: saveH1cat(aa, h1mass, halocat, fname=catname, ofolder=ofolder)
    #measure power
    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
    
    pkm = FFTPower(dm/dm.cmean(), mode='1d').power
    k, pkm = pkm['k'], pkm['power']

    ##H1
    print("H1")
    h1mesh = pm.paint(hpos, mass=h1mass)    
    pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
    pkh1m = FFTPower(h1mesh/h1mesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']
    ###Halos #Uncomment the following to estimate halo power and use it
    #hpmesh = pm.paint(hpos)
    #hmesh = pm.paint(hpos, mass=hmass)
    #pkh = FFTPower(hmesh/hmesh.cmean(), mode='1d').power['power']
    #pkhp = FFTPower(hpmesh/hpmesh.cmean(), mode='1d').power['power']
    #pkhm = FFTPower(hmesh/hmesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']
    #pkhpm = FFTPower(hpmesh/hpmesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']

    #Bias
    b1h1 = pkh1m/pkm
    b1h1sq = pkh1/pkm

    if saveb:
        np.savetxt(ofolder+bname, np.stack((k, b1h1, b1h1sq**0.5), axis=1), 
                                           header='k, pkh1xm/pkm, pkh1/pkm^0.5')

    return k, b1h1, b1h1sq


    
    

if __name__=="__main__":
    #bs, nc, sim are set in the global parameters at the top

    print('Starting')
    fig, ax = plt.subplots(1, 2, figsize = (9, 4))

    for aa in aafiles:
        ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)        
        print('Redshift = %0.2f'%(1/aa-1))
        k, b1, b1sq = fiddlebias(aa, ofolder=ofolder)
        ax[0].plot(k, b1, label='%0.2f'%(1/aa-1))
        ax[1].plot(k, b1sq**0.5)
    
    ax[0].legend()
    for axis in ax: 
        axis.set_xscale('log')
        axis.set_ylim(1, 10)
        #axis.grid(which='both', lw=0.3)
    fig.savefig('./figs/testbias.png')
