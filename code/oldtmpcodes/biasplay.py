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



def HI_masscutfiddle(mhalo,aa, mcutf=1.0):
    """Makes a 21cm "mass" from a box of halo masses.
    Use mcutf to fiddle with Mcut
    """
    print('Assigning weights')
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = (1+2*zz)/(2+2*zz)
    norm = 2e9*np.exp(-1.9*zp1+0.07*zp1**2)

    mcut= 1e10*(6.11-1.99*zp1+0.165*zp1**2)*mcutf
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm

    return(mHI)
    



def fiddlebias(aa, saveb=False,  bname='h1bias.txt', ofolder=None):
    #Fiddling bias for halos. Deprecated. 
    print('Read in catalogs')
    halocat = readincatalog(aa, matter=False)
    hpos = halocat['Position']
    hmass = halocat['Mass']
    h1mass = dohod.HI_mass(hmass, aa)
    dm = BigFileMesh(project + sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc, '1').paint()
    if ofolder is None: ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    
    #measure power
    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
    
    pkm = FFTPower(dm/dm.cmean(), mode='1d').power
    k, pkm = pkm['k'], pkm['power']

    ##H1
    print("H1")
    h1mesh = pm.paint(hpos, mass=h1mass)    
    pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
    pkh1m = FFTPower(h1mesh/h1mesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']

    #Bias
    b1h1 = pkh1m/pkm
    b1h1sq = pkh1/pkm

    if saveb:
        np.savetxt(ofolder+bname, np.stack((k, b1h1, b1h1sq**0.5), axis=1), 
                                           header='k, pkh1xm/pkm, pkh1/pkm^0.5')

    return k, b1h1, b1h1sq



def fiddlebiasgal(aa, suff, nc=nc, mcfv=[1.], saveb=False, bname='h1bias', ofolder=None):
    '''Fiddle bias for galaxies'''

    if ofolder is None: ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    print('Read in catalogs')
    cencat = BigFileCatalog(project + sim + '/fastpm_%0.4f/cencat'%aa)
    satcat = BigFileCatalog(project + sim + '/fastpm_%0.4f/satcat'%aa+suff)

    cpos, spos = cencat['Position'], satcat['Position']
    cmass, smass = cencat['Mass'], satcat['Mass']
    pos = np.concatenate((cpos, spos), axis=0)

    dm = BigFileMesh(project + sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc, '1').paint()
    pkm = FFTPower(dm/dm.cmean(), mode='1d').power
    k, pkm = pkm['k'], pkm['power']

    b1, b1sq = np.zeros((k.size, len(mcfv))), np.zeros((k.size, len(mcfv)))

    for imc, mcf in enumerate(mcfv):
        print(mcf)
        ch1mass =  HI_masscutfiddle(cmass, aa, mcutf=mcf)   
        sh1mass =  HI_masscutfiddle(smass, aa, mcutf=mcf)   
        h1mass = np.concatenate((ch1mass, sh1mass), axis=0)    
        #
        h1mesh = pm.paint(pos, mass=h1mass)    
        pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
        pkh1m = FFTPower(h1mesh/h1mesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']
        #Bias
        b1[:, imc] = pkh1m/pkm
        b1sq[:, imc] = pkh1/pkm

    np.savetxt(ofolder+bname+'auto'+suff+'.txt', np.concatenate((k.reshape(-1, 1), b1sq**0.5), axis=1), 
                                           header='mcut factors = %s\nk, pkh1xm/pkm, pkh1/pkm^0.5'%mcfv)
    np.savetxt(ofolder+bname+'cross'+suff+'.txt', np.concatenate((k.reshape(-1, 1), b1), axis=1), 
                                           header='mcut factors = %s\nk, pkh1xm/pkm, pkh1mx/pkm'%mcfv)

    return k, b1, b1sq


    
    

if __name__=="__main__":
    #bs, nc, sim are set in the global parameters at the top

    print('Starting')


    for satsuff in ['-m1_5p0min-alpha_0p9', '-m1_8p0min-alpha_0p9']:

        fig, ax = plt.subplots(3, 3, figsize=(12, 12))

        for ia, aa in enumerate(aafiles):
            mcfv = [0.5, 0.8, 1.0, 1.2, 1.5, 2]
            print('Redshift = %0.2f'%(zzfiles[ia]))
            ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)        
            k, b1, b1sq = fiddlebiasgal(aa, mcfv=mcfv, nc=256, suff=satsuff, ofolder=ofolder, saveb=False)

            axis=ax.flatten()[ia]
            axis.plot(mcfv, b1[1:5].mean(axis=0), 'C%do'%0, label='cross')
            axis.plot(mcfv, (b1sq**0.5)[1:5].mean(axis=0), 'C%d*'%0, label='auto')
            axis.set_title('z = %0.2f'%zzfiles[ia])
            axis.grid(which='both', lw=0.3)

        ax[0, 0].legend()
        for axis in ax[:, 0]:axis.set_ylabel('b$_1$$')
        for axis in ax[-1, :]:axis.set_xlabel('M_{cut} factor$')
        fig.savefig('./figs/biasmcf-%s.png'%(satsuff))

#
#    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
#    for i, aa in enumerate(aafiles):
#        ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)        
#        print('Redshift = %0.2f'%(1/aa-1))
#        k, b1, b1sq = fiddlebiasgal(aa, nc=256, suff=satsuff, ofolder=ofolder, saveb=False)
#        ax[0].plot(k, b1, 'C%d'%i, lw=1.5)
#        ax[0].plot(k, b1sq**0.5, 'C%d--'%i, lw=2)
#        ax[1].plot(1/aa-1, b1[1:5].mean(), 'C%do'%i, label='%0.2f'%(1/aa-1))
#        ax[1].plot(1/aa-1, (b1sq**0.5)[1:5].mean(), 'C%d*'%i)
#
#    for axis in ax: 
#        axis.legend()
#        ax[0].set_xscale('log')
#        axis.set_ylim(1, 5)
#        #axis.grid(which='both', lw=0.3)
#    fig.savefig('./figs/bias%s.png'%satsuff
