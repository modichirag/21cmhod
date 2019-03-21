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

#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[4:]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
#ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs,nc,ncsim = 1024, 1024, 10240
#sim,prefix  = 'highres/%d-9100-fixed'%ncsim, 'highres'

# It's useful to have my rank for printing...                                                                                                                                                                                                                                             
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm



def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
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


def measurepk(nc=nc, dpath=myscratch):
    '''plot the power spectrum of halos and H1 on 'nc' grid'''

    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])

    for i, aa in enumerate(aafiles):
        fout = "../data/outputs/halos/HI_pks_1d_{:6.4f}.txt".format(aa))
        zz = zzfiles[i]
        print('redshift = ', zz)
        dm = BigFileMesh(dpath + sim + '/fastpm_%0.4f/'%aa + '/dmesh_N%04d'%nc, '1').paint()
        rank = dm.comm.rank
        halos = BigFileCatalog(dpath + sim + '/fastpm_%0.4f/halocat/'%aa)
        hmass = halos["Mass"].compute()
        hpos = halos['Position'].compute()

#        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
#        totweight        = comm.allreduce(rankweight)
#        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
#        layout = pm.decompose(hpos)
#        
        print("paint")
        hpmesh = pm.paint(hpos, layout=layout)
        hmesh = pm.paint(hpos, mass=hmass, layout=layout)
        print(rank, dm.cmean(), hmesh.cmean(), hpos.cmean())
        print("measure powers")
        pkm = FFTPower(dm/dm.cmean(), mode='1d').power
        pkh = FFTPower(hmesh/hmesh.cmean(), mode='1d').power
        pkhp = FFTPower(hpmesh/hpmesh.cmean(), mode='1d').power
        pkhm = FFTPower(hmesh/hmesh.cmean(), second=dm/dm.cmean(), mode='1d').power
        pkhpm = FFTPower(hpmesh/hpmesh.cmean(), second=dm/dm.cmean(), mode='1d').power



        def savebinned(path, binstat, header):
            if halos.comm.rank == 0:
                k, p, modes = binstat['k'].real, binstat['power'].real, binstat['modes'].real
                np.savetxt(path, np.stack((k, p, modes), axis=1), header=header)

#        inb = 6 
#        biases = [(pkh[1:inb]['power']/pkm[1:inb]['power']).mean()**0.5, (pkhp[1:inb]['power']/pkm[1:inb]['power']).mean()**0.5, (pkhm[1:inb]['power']/pkm[1:inb]['power']).mean(), (pkhpm[1:inb]['power']/pkm[1:inb]['power']).mean()]
#        #print(biases)
#
            
        if rank == 0:
            savebinned(ofolder+'pkhpos.txt', pkhp, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhmass.txt', pkh, header='k, P(k), Nmodes')  
            savebinned(ofolder+'pkm.txt', pkm, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhposxm.txt', pkhpm, header='k, P(k), Nmodes')
            savebinned(ofolder+'pkhmassxm.txt', pkhm, header='k, P(k), Nmodes')



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
        pass
        #print(aa)
        #readincatalog(aa=aa)
        #assignH1mass(aa=aa)
        #savecatalogmesh(bs=bs, nc=256, aa=aa)
    #edges = np.logspace(np.log10(0.5), np.log10(20), 10)
    # use 1000 particles up to (20 Mpc/h) ** 3 volume;
    # looks good enough?
    #measurexi(N=1000, edges=edges)
    #make_galcat(aa=0.2000)
