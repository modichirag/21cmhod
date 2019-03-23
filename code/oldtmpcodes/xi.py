import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, MultipleSpeciesCatalog, transform
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.lab import SimulationBox2PCF, FFTCorr
import os

import sys
sys.path.append('./utils')
import tools, dohod             # 
from time import time

#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
scratch2 = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
aafiles = aafiles[2:]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#Maybe set up to take in as args file?
bs, nc = 256, 256
ncsim, sim, prefix = 256, 'lowres/%d-9100-fixed'%256, 'lowres'
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
satsuff = '-m1_5p0min-alpha_0p8'



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


# This should be imported once from a "central" place.
def HI_hod(mhalo,aa):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = 1.0
    alp = (1+2*zz)/(2+2*zz)
    mcut= 1e9*( 1.8 + 15*(3*aa)**8 )
    norm= 3e5*(1+(3.5/zz)**6)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #



def measurexi(N, edges):
    '''plot the power spectrum of halos and H1 with subsampling'''

    for i, aa in enumerate(aafiles):

        dm = BigFileCatalog(scratch  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
        halos = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)
        h1 = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)
        halos['Weight'] = halos['Mass']
        h1['Weight'] = HI_hod(halos['Mass'], aa)
        dm['Weight'] = np.ones(dm.size)

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

        #halos = subsampler(halos, rng, N, edges.max())
        #h1 = subsampler(h1, rng, N, edges.max())
        dm = subsampler(dm, rng, N, edges.max())


        if rank == 0 : print("Correlation function for edges :\n", edges)
        start=time()
        #xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter = ', end-start)
        start=end
        #xih = SimulationBox2PCF('1d',  data1=halos, edges=edges)
        end=time()
        if rank == 0 : print('Time for halos = ', end-start)
        start=end
        #ximxh = SimulationBox2PCF('1d',  data1=halos, data2=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter x halos = ', end-start)
        start=end

        #Others mass weighted
        #xihmass = SimulationBox2PCF('1d',  data1=halos, weight='Weight', edges=edges)
        xih1mass = SimulationBox2PCF('1d',  data1=h1, weight='Weight', edges=edges)
        #ximxhmass = SimulationBox2PCF('1d',  data1=halos, data2=dm, weight='Weight', edges=edges)
        ximxh1mass = SimulationBox2PCF('1d',  data1=h1, data2=dm, weight='Weight', edges=edges)
        

        def savebinned(path, binstat, header):
            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
            if rank == 0:
                try:
                    os.makedirs(os.path.dirname(path))
                except IOError:
                    pass
                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
            
        ofolder = project + '/%s/fastpm_%0.4f/ss_cm-%d/' % (sim, aa, N)
        
        #savebinned(ofolder+'xihpos.txt', xih, header='r, xi(r)')
        #savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
        #savebinned(ofolder+'xihmass.txt', xihmass, header='r, xi(r)')
        savebinned(ofolder+'xih1mass.txt', xih1mass, header='r, xi(r)')
        #savebinned(ofolder+'ximxhmass.txt', ximxhmass, header='r, xi(r)')
        savebinned(ofolder+'ximxh1mass.txt', ximxh1mass, header='r, xi(r)')









def measurexi(N, edges):
    '''plot the power spectrum of halos and H1 with subsampling'''

    for i, aa in enumerate(aafiles):

        dm = BigFileCatalog(scratch  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
        halos = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)
        h1 = BigFileCatalog(project + sim + '/fastpm_%0.4f/halocat/'%aa)
        halos['Weight'] = halos['Mass']
        h1['Weight'] = HI_hod(halos['Mass'], aa)
        dm['Weight'] = np.ones(dm.size)

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

        #halos = subsampler(halos, rng, N, edges.max())
        #h1 = subsampler(h1, rng, N, edges.max())
        dm = subsampler(dm, rng, N, edges.max())


        if rank == 0 : print("Correlation function for edges :\n", edges)
        start=time()
        #xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter = ', end-start)
        start=end
        #xih = SimulationBox2PCF('1d',  data1=halos, edges=edges)
        end=time()
        if rank == 0 : print('Time for halos = ', end-start)
        start=end
        #ximxh = SimulationBox2PCF('1d',  data1=halos, data2=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter x halos = ', end-start)
        start=end

        #Others mass weighted
        #xihmass = SimulationBox2PCF('1d',  data1=halos, weight='Weight', edges=edges)
        xih1mass = SimulationBox2PCF('1d',  data1=h1, weight='Weight', edges=edges)
        #ximxhmass = SimulationBox2PCF('1d',  data1=halos, data2=dm, weight='Weight', edges=edges)
        ximxh1mass = SimulationBox2PCF('1d',  data1=h1, data2=dm, weight='Weight', edges=edges)
        

        def savebinned(path, binstat, header):
            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
            if rank == 0:
                try:
                    os.makedirs(os.path.dirname(path))
                except IOError:
                    pass
                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
            
        ofolder = project + '/%s/fastpm_%0.4f/ss_cm-%d/' % (sim, aa, N)
        
        #savebinned(ofolder+'xihpos.txt', xih, header='r, xi(r)')
        #savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
        #savebinned(ofolder+'xihmass.txt', xihmass, header='r, xi(r)')
        savebinned(ofolder+'xih1mass.txt', xih1mass, header='r, xi(r)')
        #savebinned(ofolder+'ximxhmass.txt', ximxhmass, header='r, xi(r)')
        savebinned(ofolder+'ximxh1mass.txt', ximxh1mass, header='r, xi(r)')





def measurexigal(N, edges):
    '''plot the power spectrum of halos and H1 with subsampling'''

    for i, aa in enumerate(aafiles):

        dm = BigFileCatalog(scratch  + sim + '/fastpm_%0.4f/'%aa , dataset='1')
        cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat'%aa)
        satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+satsuff)
        cencat['HImass'] = HI_hod(cencat['Mass'],aa)
        satcat['HImass'] = HI_hod(satcat['Mass'],aa)
        cencat['Weight'] = cencat['HImass']
        satcat['Weight'] = satcat['HImass']
        dm['Weight'] = np.ones(dm.size)

        for cat in [dm, cencat, satcat]: # nbodykit bug in SimBox2PCF that asserts boxsize
            cat.attrs['BoxSize'] = np.broadcast_to(cat.attrs['BoxSize'], 3)

        rng = np.random.RandomState(dm.comm.rank)
        rank = dm.comm.rank

        zz = zzfiles[i]
        if rank == 0 : 
            print('redshift = ', zz)
            print('Number of dm particles = ', dm.csize)
            print('Number of halos particles = ', cencat.csize+satcat.csize)
            
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

        #h1 = subsampler(allcat, rng, N, edges.max())
        dm = subsampler(dm, rng, N, edges.max())
        #cencat = subsampler(cencat, rng, N, edges.max())
        #satcat = subsampler(satcat, rng, N, edges.max())
        h1 = transform.ConcatenateSources(cencat, satcat)


        if rank == 0 : print("Correlation function for edges :\n", edges)
        start=time()
        xim = SimulationBox2PCF('1d',  data1=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter = ', end-start)
        start=end
        xigal_h1 = SimulationBox2PCF('1d',  data1=h1, edges=edges)
        end=time()
        if rank == 0 : print('Time for halos = ', end-start)
        start=end
        xigal_mxh1 = SimulationBox2PCF('1d',  data1=h1, data2=dm, edges=edges)
        end=time()
        if rank == 0 : print('Time for matter x halos = ', end-start)
        

        def savebinned(path, binstat, header):
            r, xi = binstat.corr['r'].real, binstat.corr['corr'].real
            if rank == 0:
                try:
                    os.makedirs(os.path.dirname(path))
                except IOError:
                    pass
                np.savetxt(path, np.stack((r, xi), axis=1), header=header)
            
        ofolder = project + '/%s/fastpm_%0.4f/ss_cm-%d/' % (sim, aa, N)

        savebinned(ofolder+'ximatter.txt', xim, header='r, xi(r)')
        savebinned(ofolder+'xigal_h1.txt', xigal_h1, header='r, xi(r)')
        savebinned(ofolder+'xigal_mxh1.txt', xigal_mxh1, header='r, xi(r)')


    

if __name__=="__main__":

    edges = np.logspace(np.log10(0.5), np.log10(20), 10)
    # use 1000 particles up to (20 Mpc/h) ** 3 volume;
    # looks good enough?
    measurexigal(N=10000, edges=edges)
    measurexi(N=10000, edges=edges)
