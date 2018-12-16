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
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[4:]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
#ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'



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



def make_galcat(aa,save=False, seed=3333):
    '''do HOD with Zheng model using nbodykit'''
    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)
    halocat = readincatalog(aa)
    hmass = halocat['Mass'].compute()

    #Do hod
    ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    
    dMda = -0.1
    mcut, m1 = mkhodp_linear(aa, apiv=0.6667, hod_dMda=-0.1, mcut_apiv=10**13.35, m1_apiv=10 ** 12.80)
    print('Mcut, m1 : ', np.log10(mcut), np.log10(m1))
    print('Halomass0 : ',np.log10(halocat['Mass'].compute()[1]))
    (ncen, cpos, cvel), (nsat, spos, svel) = hod(seed,halocat['Mass'].compute(), halocat['Position'].compute(), halocat['Velocity'].compute(),\
                        conc=7, rvir=3, vdisp=1100, mcut=mcut, m1=m1, sigma=0.25, \
                        kappa=1.0, alpha=0.8,vcen=0, vsat=0.5)


    hid = np.repeat(range(len(hmass)), ncen).astype(int)
    print('Number of centrals = ', ncen.sum())
    #Assign mass to centrals
    cmass = hmass[hid]
    #Assign HI mass
    ch1mass = dohod.HI_mass(cmass, aa)
    cencat = ArrayCatalog({'Position':cpos, 'Velocity':cvel, 'Mass':cmass, 'H1mass':ch1mass, 'HaloID':hid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])


    hid = np.repeat(range(len(hmass)), nsat).astype(int)
    print('Number of satellites = ', nsat.sum())
    np.random.seed(seed)
    #Assign mass to satellites
    smass = np.random.uniform(size=hid.size)
    smass = mcut*0.1*hmass[hid]/3. / ((1-smass)*hmass[hid]/3. + smass*mcut*0.1)
    #Assign HI mass
    sh1mass = dohod.HI_mass(smass, aa)

    satcat = ArrayCatalog({'Position':spos, 'Velocity':svel, 'Mass':smass, 'H1mass':sh1mass, 'HaloID':hid}, 
                          BoxSize=halocat.attrs['BoxSize'], Nmesh=halocat.attrs['NC'])
    if save:
        colsave = [cols for cols in cencat.columns]
        cencat.save(ofolder+'cencat', colsave)
        colsave = [cols for cols in satcat.columns]
        satcat.save(ofolder+'satcat', colsave)
#

if __name__=="__main__":

    for aa in aafiles:
        make_galcat(aa=aa, save=True)
