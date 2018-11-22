import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
import sys
sys.path.append('./utils')
import tools, dohod

#Global, fixed things
dpath = '/global/cscratch1/sd/yfeng1/m3127/lowres/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
mp = 8.57799 * 1e10
sim = '%d-9100-fixed'%nc
pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])


def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(dpath + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(dpath + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat


def make_galcat(aa,save=True):

    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)
    halocat = readincatalog(aa)
    #Do hod
    ofolder = '../data/%s/fastpm_%0.4f/'%(sim, aa)
    galcat = dohod.make_galcat(halocat, ofolder, z=zz, pm=pm)
    if save:
        colsave = [cols for cols in galcat.columns]
        galcat.save(ofolder+'galcat', colsave)
        print('Galaxies saved at path\n%s'%ofolder)


def assignH1mass(aa, save=True):

    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)

    halocat = readincatalog(aa)
    #Do hod
    ofolder = '../data/%s/fastpm_%0.4f/'%(sim, aa)
    halocat['H1mass'] = dohod.assignH1mass(halocat, z=zz)
    if save:
        colsave = [cols for cols in halocat.columns]
        halocat.save(ofolder+'halocat', colsave)
        print('Halos saved at path\n%s'%ofolder)
    


if __name__=="__main__":

    make_galcat(aa=0.2000)
    assignH1mass(aa=0.2000)
