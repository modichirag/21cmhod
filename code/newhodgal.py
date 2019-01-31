import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
import os

import hod

import sys
sys.path.append('./utils')
import tools, dohod             # 
from time import time

#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'

cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
print(cosmo)
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[:2]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
#bs, nc, ncsim, sim, prefix = 256, 256, 256, 'lowres/%d-9100-fixed'%256, 'lowres'
bs, nc, ncsim, sim, prefix = 256, 256, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'



def readincatalog(aa):

    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    return halocat



def make_galcat(aa, mmin, m1, alpha=1, censuff=None, satsuff=None, ofolder=None, seed=3333):
    '''Assign 0s to 
    '''
    zz = tools.atoz(aa)
    halocat = readincatalog(aa)
    rank = halocat.comm.rank
    if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%zz)

    hmass = halocat['Mass'].compute()
    hpos = halocat['Position'].compute()
    hvel = halocat['Velocity'].compute()
    rvir = HaloRadius(hmass, cosmo, 1/aa-1).compute()/aa
    vdisp = HaloVelocityDispersion(hmass, cosmo, 1/aa-1).compute()
    ghid = halocat.Index.compute()

    print('In rank = %d, Catalog size = '%rank, hmass.size)
    #Do hod    
    start = time()
    ncen = np.ones_like(hmass)
    nsat = hod.nsat(hmass, m0=0, m1=m1, alpha=alpha).astype(int)
    
    #Centrals
    cpos, cvel, gchid = hpos, hvel, ghid
    spos, svel, shid = hod.mksat(nsat, hindex=ghid, pos=hpos, vel=hvel, vdisp=vdisp, conc=7, rvir=rvir, vsat=0.5, seed=seed)
    gshid = ghid[shid]
    svelh1 = svel*2/3 + cvel[shid]/3.

    smmax = hmass[shid]/10.
    smmin = np.ones_like(smmax)*mmin
    mask = smmin > smmax/3. #Some fudge that should be discussed
    smmin[mask] = smmax[mask]/3.
    smass = hod.get_msat(hmass[shid], smmax, smmin, alpha)


    #get satellite mass in every halo
    sathmass = np.zeros_like(hmass)
    unipos, unicount = np.unique(shid, return_inverse=True)
    for i in range(unicount.size):
        sathmass[unipos[unicount[i]]] +=  smass[i]
    
    cmass = hmass - sathmass    # assign remaining mass in centrals

    print('In rank = %d, Time taken = '%rank, time()-start)
    print('In rank = %d, Number of centrals & satellites = '%rank, ncen.sum(), nsat.sum())
    print('In rank = %d, Satellite occupancy: Max and mean = '%rank, nsat.max(), nsat.mean())
    #
    #Save
    cencat = ArrayCatalog({'Position':cpos, 'Velocity':cvel, 'Mass':cmass,  'HaloID':gchid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    if censuff is not None:
        colsave = [cols for cols in cencat.columns]
        cencat.save(ofolder+'cencat'+censuff, colsave)
    

    satcat = ArrayCatalog({'Position':spos, 'Velocity':svel, 'Velocity_HI':svelh1, 'Mass':smass,  'HaloID':gshid, 'LocalHaloID':shid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    if satsuff is not None:
        colsave = [cols for cols in satcat.columns]
        satcat.save(ofolder+'satcat'+satsuff, colsave)

#

if __name__=="__main__":

    for aa in aafiles[:]:

        ofolder = myscratch + '/%s/fastpm_%0.4f/'%(sim, aa)

        #sat hod : N = ((M_h-\kappa*mcut)/m1)**alpha
        zz = 1/aa-1

        #mmin = 10**(11-0.4*np.array(zz)) #This is already 1/10th of mcut
        #Update mmin with the current mcut from draft
        mmin = 1e9*( 1.8 + 15*(3*aa)**8 ) * 0.1 #mcut * 0.1, 0.1 being mmin


        alpha = 0.8
        for m1fac in [5.0]:
            censuff ='-m1_%dp%dmin-alpha_0p8-v2'%(int(m1fac), (m1fac*10)%10)
            satsuff ='-m1_%dp%dmin-alpha_0p8-v2'%(int(m1fac), (m1fac*10)%10)

            m1 = m1fac*mmin
            make_galcat(aa=aa, mmin=mmin, mcutc=mcutc, m1=m1m, sigma=sigma, kappa=kappa, alpha=alpha, 
                        censuff=censuff, satsuff=satsuff, ofolder=ofolder)

    



