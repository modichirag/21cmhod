import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
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
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[:2]
#aafiles = [0.2222]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
#bs, nc = 256, 256
#ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
#ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
bs, nc = 1024, 1024
ncsim, sim, prefix = 10240, 'highres/%d-9100-fixed'%10240, 'highres'



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
    print('In rank = %d, Catalog size = '%rank, hmass.size)
    #Do hod
    ofolder = myscratch + '/%s/fastpm_%0.4f/'%(sim, aa)
    try:
        os.makedirs(os.path.dirname(ofolder))
    except IOError:
        pass
    
    #hoddMda = -0.1
    #mcut, m1 = mkhodp_linear(aa, apiv=0.6667, hod_dMda=hoddMda, mcut_apiv=10**13.35, m1_apiv=10 ** 12.80)
    
    #print('Mcut, m1 : ', np.log10(mcutc), np.log10(m1))
    #print('kappa  = ', kappa)
    #print('Halomass0 : ',np.log10(halocat['Mass'].compute()[1]))
    #print('Number of halos = ', hmass.size)
    #print('Number of halos above mMin = ', (hmass > mminh1).sum())
    #print('Number of halos above mcut = ', (hmass > mcutc).sum())
    start = time()
    (ncen, cpos, cvel), (nsat, spos, svel) = hod(seed*rank, hmass, halocat['Position'].compute(), halocat['Velocity'].compute(),\
                        conc=7, rvir=3, vdisp=1100, mcut=mcutc, m1=m1, sigma=0.25, \
                        kappa=kappa, alpha=alpha,vcen=0, vsat=0.5)

    print('In rank = %d, Time taken = '%rank, time()-start)
    print('In rank = %d, Number of centrals & satellites = '%rank, ncen.sum(), nsat.sum())
    print('In rank = %d, Satellite occupancy: Max and mean = '%rank, nsat.max(), nsat.mean())
    #
    #Assign mass to centrals
    hid = np.repeat(range(len(hmass)), ncen).astype(int)
    cmass = hmass[hid]
    #Assign HI mass
    #ch1mass = dohod.HI_mass(cmass, aa)
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
    mask = mmin > hmass[hid]/10.
    mmin[mask] = hmass[hid][mask]/10.
    smass = mmin * mmax / ((1-smass)*mmax + smass*mmin)
    #smass[smass>mmax] = 0
    #Assign HI mass
    #sh1mass = dohod.HI_mass(smass, aa)

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
        #mmin /= 10
        alpha = 0.8
        for m1 in [5.0]:
            satsuff='-m1_%dp%dmin-alpha_0p8-16node'%(int(m1), (m1*10)%10)
            #print('\n', mmin, m1, satsuff, '\n')
            m1m = m1*mmin
            make_galcat(aa=aa, mmin=mmin, mcutc=mcutc, m1=m1m, sigma=sigma, kappa=kappa, alpha=alpha, censuff='-16node', satsuff=satsuff)

#        alpha = 0.8
#        for m1 in [5, 10, 20]:
#            satsuff='-mmin0p1_m1_%dp%dmin-alpha_0p8'%(int(m1), (m1*10)%10)
#            print('\n', mmin, m1, satsuff, '\n')
#            m1m = m1*mmin
#            make_galcat(aa=aa, mmin=mmin, mcutc=mcutc, m1=m1m, sigma=sigma, kappa=kappa, alpha=alpha, censuff=None, satsuff=satsuff)
#
