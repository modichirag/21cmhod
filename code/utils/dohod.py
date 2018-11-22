import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from pmesh.pm import ParticleMesh

#from nbodykit.cosmology import Planck15, EHPower
from nbodykit.lab import ArrayCatalog
from nbodykit.lab import KDDensity, BigFileCatalog
from nbodykit.lab import cosmology, HaloCatalog, Zheng07Model
import os, sys, json
import icdfhod_sampling as icdf

hodparamsdef = {'alpha':0.775668, 'logMmin':13.053998, 'logM1':14.3, 'logM0':13.6176, 'sigma_logM':0.397894}
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}

def populategalaxies(halos, zz=0, hodparams=None, cosmo=None, seedhod=43):
    '''Read halo file or pos/mas arrays and grid and smooth them after scattering or matching abundance or both    
    The path should have folder FOF in it                                
    '''

    if hodparams is None: hodparams = hodparamsdef
    if cosmo is None: cosmo = cosmology.Cosmology(Omega_b = cosmodef['omegab'], Omega_cdm = cosmodef['omegam'], h=cosmodef['h'])

    #Test if factors are normalized 
#    if halos['Velocity'].compute().max() < 1e2: halos['Velocity'] = halos['Velocity']*100
#    if halos['Mass'].compute().max() < 1e8: halos['Mass'] = halos['Mass'] * 1e10

    # create the halo catalog
    halocat = HaloCatalog(halos, cosmo=cosmo, redshift=zz, mdef='vir') #mdef can be m200,c200,vir

    # populate halos with galaxies 
    galcat = halocat.populate(Zheng07Model, seed=seedhod, **hodparams, BoxSize=halos.attrs['BoxSize'])

    print('Satellite fraction  = ', galcat.attrs['fsat'])
    return galcat





def make_galcat(fofcat, ofolder='', z=0, sortcen=False, sortsat=False, sigc=0.212, mstarc=1e12, hodparams=None, pm=None):
    #initiate
    if pm is None: pm = ParticleMesh(BoxSize=8, Nmesh=(32, 32, 32), dtype='f8')

    try: os.makedirs(ofolder)
    except: pass
    
    if hodparams is None: hodparams = hodparamsdef

    with open(ofolder + 'hodparams.json', 'w') as fp:
            json.dump(hodparams, fp, sort_keys=True, indent=4)

    galcat = populategalaxies(halos=fofcat, zz=z, hodparams=hodparams)

    print('Galaxies populated')
    return galcat


    ### Assigning masses to galaxies below this
    ## We don't need it right now
#    Mthresh = icdf.mstarsat(fofcat['Mass'].compute()[-1])
#    print('Mthresh = %0.2e'%Mthresh)
#    nmbins = 50
#    #cenpars = {sortcen=False, #EDITING HERE
#    galmass = icdf.assigngalaxymass(galcat, fofcat['Mass'].compute(), nmbins=nmbins, Mthresh=Mthresh, 
#                                    sortcen=sortcen, sortsat=sortsat, sigc=sigc, mstarc=mstarc)
#    print('Galaxy masses assigned')
#
#    galcat['Mass'] = galmass
#
#    colsave = [cols for cols in galcat.columns]
#    fname = 'galcat'
#    if sortcen: fname+='-sortcen'
#    if sortsat: fname+='-sortsat'
#    if sigc !=0.212: fname += '-sigc%02d'%(sigc*100)
#    print('Data saved at %s'%(ofolder + fname))
#    galcat.save(ofolder+fname, colsave)
#

def Az(z):
    '''Taken from table 6 of arxiv:1804.09180'''
    zz = np.array((0, 1, 2, 3, 4, 5))
    yy = np.ones_like(zz)
    return np.interp(z, zz, yy)

def alphaz(z):
    '''Taken from table 6 of arxiv:1804.09180'''
    zz = np.array((0, 1, 2, 3, 4, 5))
    yy = np.array((0.49, 0.76, 0.80, 0.95, 0.94, 0.90))
    return np.interp(z, zz, yy)

def M0z(z):
    '''Taken from table 6 of arxiv:1804.09180'''
    zz = np.array((0, 1, 2, 3, 4, 5))
    yy = np.array((2.1e9, 4.6e8, 4.9e8, 9.2e7, 6.4e7, 9.5e7))
    return np.interp(z, zz, yy)

def Mminz(z):
    '''Taken from table 6 of arxiv:1804.09180'''
    zz = np.array((0, 1, 2, 3, 4, 5))
    yy = np.array((5.2e10, 2.6e10, 2.1e10, 4.8e9, 2.1e9, 1.9e9))
    return np.interp(z, zz, yy)


def assignH1mass(halos, z=0):
    '''Assign H1 mass based on expresion in overleaf Eq. 4.1'''
    A = Az(z)
    mmin = Mminz(z)
    m0 = M0z(z)
    alpha = alphaz(z)
    
    mass = halos['Mass'].compute()
    mh1 = A*mass**alpha * np.exp(-mmin/mass)
    
    return mh1
