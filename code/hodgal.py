import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
import os

from simplehod import hod, mkhodp_linear

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
bs, nc, ncsim, sim, prefix = 256, 256, 256, 'lowres/%d-9100-fixed'%256, 'lowres'
bs, nc, ncsim, sim, prefix = 256, 256, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'



def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    #print('Mass of particle is = %0.2e'%mp)
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat



def make_galcat(aa, mmin, mcutc, m1, sigma=0.25, kappa=1, alpha=1, censuff=None, satsuff=None, seed=3333):
    '''do HOD with Zheng model using nbodykit'''
    zz = tools.atoz(aa)
    halocat = readincatalog(aa)
    rank = halocat.comm.rank
    if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%zz)

    hmass = halocat['Mass'].compute()
    hpos = halocat['Position'].compute()
    hvel = halocat['Velocity'].compute()
    rvir = HaloRadius(cmass, cosmo, 1/aa-1).compute()/aa
    vdisp = HaloVelocityDispersion(cmass, cosmo, 1/aa-1).compute()

    print('In rank = %d, Catalog size = '%rank, hmass.size)
    #Do hod
    ofolder = myscratch + '/%s/fastpm_%0.4f/'%(sim, aa)
    try:
        os.makedirs(os.path.dirname(ofolder))
    except IOError:
        pass
    
    start = time()
    #(ncen, cpos, cvel), (nsat, spos, svel) = hod(seed*rank, hmass, halocat['Position'].compute(), halocat['Velocity'].compute(),\
    #                    conc=7, rvir=3, vdisp=1100, mcut=mcutc, m1=m1, sigma=0.25, \
    #                    kappa=kappa, alpha=alpha,vcen=0, vsat=0.5)
    
    (ncen, cpos, cvel), (nsat, spos, svel) = hod(seed*rank, hmass, hpos, hvel, 
                        conc=7, rvir=rvir, vdisp=vdisp, mcut=mcutc, m1=m1, sigma=0.25, \
                        kappa=kappa, alpha=alpha, vcen=0, vsat=0.5)

    print('In rank = %d, Time taken = '%rank, time()-start)
    print('In rank = %d, Number of centrals & satellites = '%rank, ncen.sum(), nsat.sum())
    print('In rank = %d, Satellite occupancy: Max and mean = '%rank, nsat.max(), nsat.mean())
    #
    #Assign mass to centrals
    hid = np.repeat(range(len(hmass)), ncen).astype(int)
    cmass = hmass[hid]
    cencat = ArrayCatalog({'Position':cpos, 'Velocity':cvel, 'Mass':cmass,  'HaloID':hid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])

    if censuff is not None:
        colsave = [cols for cols in cencat.columns]
        cencat.save(ofolder+'cencat'+censuff, colsave)
    
    #
    #Assign mass to satellites
    hid = np.repeat(range(len(hmass)), nsat).astype(int)
    np.random.seed(seed*rank)

    smass = np.random.uniform(size=hid.size)
    mmax = hmass[hid]/3.
    mmin = np.ones_like(mmax)*mmin
    mask = mmin > hmass[hid]/10. #Some fudge that should be discussed
    mmin[mask] = hmass[hid][mask]/10.
    smass = mmin * mmax / ((1-smass)*mmax + smass*mmin)

    satcat = ArrayCatalog({'Position':spos, 'Velocity':svel, 'Mass':smass,  'HaloID':hid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    if satsuff is not None:
        colsave = [cols for cols in satcat.columns]
        satcat.save(ofolder+'satcat'+satsuff, colsave)

#

if __name__=="__main__":

    for aa in aafiles[:]:
        mminh1 = dohod.HI_mass(None, aa, 'mcut')
        mcutc = 1
        kappa = 0
        sigma = 0
 
        #sat hod : N = ((M_h-\kappa*mcut)/m1)**alpha
        zz = 1/aa-1
        mmin = 10**(11-0.4*np.array(zz))
        #Update mmin with the current mcut from draft
        mmin = 1e9*( 1.8 + 15*(3*aa)**8 ) * 0.1 # 0.1 is the lim from appendix
        #mmin /= 10
        alpha = 0.8
        for m1 in [5.0]:
            satsuff='-m1_%dp%dmin-alpha_0p8'%(int(m1), (m1*10)%10)
            #print('\n', mmin, m1, satsuff, '\n')
            m1m = m1*mmin
            make_galcat(aa=aa, mmin=mmin, mcutc=mcutc, m1=m1m, sigma=sigma, kappa=kappa, alpha=alpha, censuff='', satsuff=satsuff)

#
