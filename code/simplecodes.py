import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.lab import SimulationBox2PCF, FFTCorr
import os

import sys
sys.path.append('./utils')
import tools, dohod             # 
from time import time

#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='for small or big box', default='small')
parser.add_argument('-a', '--amp', help='amplitude for up/down sigma 8', default=None)
parser.add_argument('-r', '--res', help='resolution', default='high')
args = parser.parse_args()

boxsize = args.size
amp = args.amp
res = args.res

#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[4:]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
if res == 'high':
    if boxsize == 'small':
        bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
    elif boxsize == 'big':
        bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'
else:
    bs, nc, ncsim, sim, prefix = 256, 256, 256, 'lowres/%d-9100-fixed'%256, 'lowres'

if amp is not None:
    if amp == 'up' or amp == 'dn': sim = sim + '-%s'%amp
    else: print('Amplitude not understood. Should be "up" or "dn". Given : {}. Fallback to fiducial'.format(amp))

# It's useful to have my rank for printing... 

pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args)


def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratchyf + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratchyf + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    print('Mass of particle is = %0.2e'%mp)
    halocat['Mass'] = halocat['Length'] * mp
    print(halocat['Mass'][:5].compute()/1e10)
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat


def savecatalogmesh(bs, nc, aa):
    dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
    mesh = pm.paint(dmcat['Position'])
    mesh = FieldMesh(mesh)
    path = project +  sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc
    mesh.save(path, dataset='1', mode='real')
    


def make_galcat(aa,save=True):
    '''do HOD with Zheng model using nbodykit'''
    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)
    halocat = readincatalog(aa)
    #Do hod
    ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    galcat = dohod.make_galcat(halocat, ofolder, z=zz, pm=pm)
    if save:
        colsave = [cols for cols in galcat.columns]
        galcat.save(ofolder+'galcat', colsave)
        print('Galaxies saved at path\n%s'%ofolder)


def assignH1mass(aa, save=True):
    '''assign H1 masses to halos based on numbers from arxiv...'''
    zz = tools.atoz(aa)
    print('Redshift = %0.2f'%zz)

    halocat = readincatalog(aa)
    #Do hod
    ofolder = project + '/%s/fastpm_%0.4f/'%(sim, aa)
    halocat['H1mass'] = dohod.assignH1mass(halocat, z=zz)


    if save:
        colsave = [cols for cols in halocat.columns]
        colsave = ['ID', 'Position', 'Mass', 'H1mass']
        print(colsave)
        halocat.save(ofolder+'halocat', colsave)
        print('Halos saved at path\n%s'%ofolder)


def measurepk(nc=nc, dpath=scratchyf):
    '''plot the power spectrum of halos on 'nc' grid'''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        if rank == 0: print('redshift = ', zz)
        if ncsim == 10240:
            dm = BigFileMesh(scratchyf+sim+'/fastpm_%0.4f/'%aa+\
                            '/1-mesh/N%04d'%nc,'').paint()
        else:  dm = BigFileMesh(project+sim+'/fastpm_%0.4f/'%aa+\
                        '/dmesh_N%04d/1/'%nc,'').paint()
        #dm = BigFileMesh(project + sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc, '1').paint()
        halos = BigFileCatalog(scratchyf + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
        mp = halos.attrs['MassTable'][1]*1e10
        if rank == 0: print('Mass of particle is = %0.2e'%mp)
        hmass = halos["Length"].compute()*mp
        hpos = halos['Position'].compute()
        layout = pm.decompose(hpos)

        if rank == 0: print("paint")
        hpmesh = pm.paint(hpos, layout=layout)
        hmesh = pm.paint(hpos, mass=hmass, layout=layout)
        print(rank, dm.cmean(), hmesh.cmean(), hpmesh.cmean())

        pkm = FFTPower(dm/dm.cmean(), mode='1d').power
        pkh = FFTPower(hmesh/hmesh.cmean(), mode='1d').power
        pkhp = FFTPower(hpmesh/hpmesh.cmean(), mode='1d').power
        pkhm = FFTPower(hmesh/hmesh.cmean(), second=dm/dm.cmean(), mode='1d').power
        pkhpm = FFTPower(hpmesh/hpmesh.cmean(), second=dm/dm.cmean(), mode='1d').power



        def savebinned(path, binstat, header):
            if halos.comm.rank == 0:
                k, p, modes = binstat['k'].real, binstat['power'].real, binstat['modes'].real
                np.savetxt(path, np.stack((k, p, modes), axis=1), header=header)

            
        ofolder = "../data/outputs/halos/{}/".format(sim)
        if rank == 0:
            print(ofolder)
            try: 
                os.makedirs(ofolder)
            except : pass
            savebinned(ofolder+'pkhp_%0.4f.txt'%aa, pkhp, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhm_%0.4f.txt'%aa, pkh, header='k, P(k), Nmodes')  
            savebinned(ofolder+'pkd_%0.4f.txt'%aa, pkm, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhpxd_%0.4f.txt'%aa, pkhpm, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhmxd_%0.4f.txt'%aa, pkhm, header='k, P(k), Nmodes')



def measurexi(N, edges):
    '''plot the power spectrum of halos and H1 with subsampling'''


    for i, aa in enumerate(aafiles):

        dm = BigFileCatalog(scratch  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
        halos = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)
        h1 = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)

        for cat in [dm, halos, h1]: # nbodykit bug in SimBox2PCF that asserts boxsize
            cat.attrs['BoxSize'] = np.broadcast_to(cat.attrs['BoxSize'], 3)

        rng = np.random.RandomState(halos.comm.rank)
        rank = dm.comm.rank

        zz = zzfiles[i]
        if rank == 0 : 
            print('redshift = ', zz)
            print('Number of dm particles = ', dm.csize)
            print('Number of halos particles = ', h1.csize)
            
        def subsampler(cat, rng, N, rmax):
            # subsample such that we have at most N particles to rmax
            nbar = (cat.csize / cat.attrs['BoxSize'].prod())
            ratio = (N / rmax ** 3) / nbar
            mask = rng.uniform(size=cat.size) < ratio
            cat1 = cat[mask]

            if rank == 0:
                print('truncating catalog from %d to %d' % (cat.csize, cat1.csize))
            return cat1

        if rank == 0 : print('Create weight array')

        halos = subsampler(halos, rng, N, edges.max())
        h1 = subsampler(h1, rng, N, edges.max())
        dm = subsampler(dm, rng, N, edges.max())

        halos['Weight'] = halos['Mass']
        h1['Weight'] = h1['H1mass']
        dm['Weight'] = np.ones(dm.size)

        if rank == 0 : print("Correlation function for edges :\n", edges)
        start=time()
        xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter = ', end-start)
        start=end
        xih = SimulationBox2PCF('1d',  data1=halos, edges=edges)
        end=time()
        if rank == 0 : print('Time for halos = ', end-start)
        start=end
        ximxh = SimulationBox2PCF('1d',  data1=halos, data2=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter x halos = ', end-start)
        start=end

        #Others mass weighted
        xihmass = SimulationBox2PCF('1d',  data1=halos, weight='Weight', edges=edges)
        xih1mass = SimulationBox2PCF('1d',  data1=h1, weight='Weight', edges=edges)
        ximxhmass = SimulationBox2PCF('1d',  data1=halos, data2=dm, weight='Weight', edges=edges)
        ximxh1mass = SimulationBox2PCF('1d',  data1=h1, data2=dm, weight='Weight', edges=edges)
        

        def savebinned(path, binstat, header):
            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
            if rank == 0:
                try:
                    os.makedirs(os.path.dirname(path))
                except IOError:
                    pass
                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
            
        ofolder = project + '/%s/fastpm_%0.4f/ss-%d/' % (sim, aa, N)
        
        savebinned(ofolder+'xihpos.txt', xih, header='r, xi(r)')
        savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
        savebinned(ofolder+'xihmass.txt', xihmass, header='r, xi(r)')
        savebinned(ofolder+'xih1mass.txt', xih1mass, header='r, xi(r)')
        savebinned(ofolder+'ximxhmass.txt', ximxhmass, header='r, xi(r)')
        savebinned(ofolder+'ximxh1mass.txt', ximxh1mass, header='r, xi(r)')


    

if __name__=="__main__":

    measurepk(nc)        
    for aa in aafiles:
        if rank == 0: print(aa)
        #readincatalog(aa=aa)
        #assignH1mass(aa=aa)
        #savecatalogmesh(bs=bs, nc=nc, aa=aa)

    #edges = np.logspace(np.log10(0.5), np.log10(20), 10)
    ## use 1000 particles up to (20 Mpc/h) ** 3 volume;
    ## looks good enough?
    #measurexi(N=1000, edges=edges)
    #make_galcat(aa=0.2000)
