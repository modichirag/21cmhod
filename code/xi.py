import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, MultipleSpeciesCatalog, transform
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.lab import SimulationBox2PCF, FFTCorr
from nbodykit     import setup_logging
import os

import sys
sys.path.append('./utils')
import tools, dohod             # 
from time import time


import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')


#
#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist = alist[-1:]

#Parameters, box size, number of mesh cells, simulation, ...
bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'


# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model & configuration to use
HImodel = HImodels.ModelA
modelname = 'ModelA'
mode = 'galaxies'
ofolder = '../data/outputs/'





def measurexi(N, edges):
    '''plot the power spectrum of halos and H1 with subsampling'''

    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:]
    if bs == 1024: outfolder = outfolder + "-big"
    outfolder += "/%s/"%modelname

    for i, aa in enumerate(alist):

        dm = BigFileCatalog(scratchyf  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
        rng = np.random.RandomState(dm.comm.rank)
        rank = dm.comm.rank

        halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10##
        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat'%aa+suff)
        satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat'%aa+suff)
        #

        HImodelz = HImodel(aa)
        los = [0,0,1]
        halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)

        for cat in [halocat, cencat, satcat]: 
            cat['Weight'] = cat['HImass'] 
        dm['Weight'] = np.ones(dm.size)
    
        for cat in [dm, halocat, cencat, satcat]: # nbodykit bug in SimBox2PCF that asserts boxsize
            cat.attrs['BoxSize'] = np.broadcast_to(cat.attrs['BoxSize'], 3)

        #
        #Combine galaxies to halos
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat['Weight'].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat['Weight'] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)

        #Subsample
        if rank == 0 : 
            print('redshift = ', 1/aa-1)
            print('Number of dm particles = ', dm.csize)
            print('Number of halos particles = ', halocat.csize)
            
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

        #halocat = subsampler(halocat, rng, N, edges.max())
        dm = subsampler(dm, rng, N, edges.max())


        if rank == 0 : print("Correlation function for edges :\n", edges)
        start=time()
        #xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter = ', end-start)

        #Others mass weighted
        
        start = time()
        xih1mass = SimulationBox2PCF('1d',  data1=halocat, weight='Weight', edges=edges)
        end=time()
        if rank == 0 : print('Time for HI = ', end-start)
        start = time()
        ximxh1mass = SimulationBox2PCF('1d',  data1=halocat, data2=dm, weight='Weight', edges=edges)
        end=time()
        if rank == 0 : print('Time for Cross = ', end-start)
        

        def savebinned(path, binstat, header):
            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
            if rank == 0:
                try:
                    os.makedirs(os.path.dirname(path))
                except IOError:
                    pass
                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
            
        #savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
        savebinned(outfolder+"xih1mass_{:6.4f}.txt".format(aa), xih1mass, header='r, xi(r)')
        savebinned(outfolder+"ximxh1mass_{:6.4f}.txt".format(aa), ximxh1mass, header='r, xi(r)')




##
##def measurexigal(N, edges):
##    '''plot the power spectrum of halos and H1 with subsampling'''
##
##    for i, aa in enumerate(aafiles):
##
##        dm = BigFileCatalog(scratch  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
##        cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat'%aa)
##        satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+satsuff)
##        cencat['HImass'] = HI_hod(cencat['Mass'],aa)
##        satcat['HImass'] = HI_hod(satcat['Mass'],aa)
##        cencat['Weight'] = cencat['HImass']
##        satcat['Weight'] = satcat['HImass']
##        dm['Weight'] = np.ones(dm.size)
##
##        for cat in [dm, cencat, satcat]: # nbodykit bug in SimBox2PCF that asserts boxsize
##            cat.attrs['BoxSize'] = np.broadcast_to(cat.attrs['BoxSize'], 3)
##
##        rng = np.random.RandomState(dm.comm.rank)
##        rank = dm.comm.rank
##
##        zz = zzfiles[i]
##        if rank == 0 : 
##            print('redshift = ', zz)
##            print('Number of dm particles = ', dm.csize)
##            print('Number of halos particles = ', cencat.csize+satcat.csize)
##            
##        def subsampler(cat, rng, N, rmax):
##            # subsample such that we have at most N particles to rmax
##            nbar = (cat.csize / cat.attrs['BoxSize'].prod())
##            ratio = (N / rmax ** 3) / nbar
##            mask = rng.uniform(size=cat.size) < ratio
##            cat1 = cat[mask]
##
##            if rank == 0:
##                print('truncating catalog from %d to %d' % (cat.csize, cat1.csize))
##            return cat1

##
##        if rank == 0 : print('Create weight array')
##
##        #h1 = subsampler(allcat, rng, N, edges.max())
##        dm = subsampler(dm, rng, N, edges.max())
##        #cencat = subsampler(cencat, rng, N, edges.max())
##        #satcat = subsampler(satcat, rng, N, edges.max())
##        h1 = transform.ConcatenateSources(cencat, satcat)
##
##
##        if rank == 0 : print("Correlation function for edges :\n", edges)
##        start=time()
##        xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
##        end=time()
##        if rank == 0 : print('Time for matter = ', end-start)
##        start=end
##        xigal_h1 = SimulationBox2PCF('1d',  data1=h1, edges=edges)
##        end=time()
##        if rank == 0 : print('Time for halos = ', end-start)
##        start=end
##        xigal_mxh1 = SimulationBox2PCF('1d',  data1=h1, data2=dm, edges=edges)
##        end=time()
##        if rank == 0 : print('Time for matter x halos = ', end-start)
##        
##
##        def savebinned(path, binstat, header):
##            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
##            if rank == 0:
##                try:
##                    os.makedirs(os.path.dirname(path))
##                except IOError:
##                    pass
##                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
##            
##        ofolder = project + '/%s/fastpm_%0.4f/ss_cm-%d/' % (sim, aa, N)
##
##        savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
##        savebinned(ofolder+'xigal_h1.txt', xigal_h1, header='r, xi(r)')
##        savebinned(ofolder+'xigal_mxh1.txt', xigal_mxh1, header='r, xi(r)')
##
##
    

if __name__=="__main__":

    edges = np.logspace(np.log10(0.5), np.log10(20), 10)
    # use 1000 particles up to (20 Mpc/h) ** 3 volume;
    # looks good enough?
    #measurexigal(N=10000, edges=edges)
    measurexi(N=10000, edges=edges)
