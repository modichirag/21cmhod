#Test the new bincout function for distributed arrays

import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
from nbodykit.utils import DistributedArray
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
aafiles = aafiles[:1]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]

#Paramteres
#bs, nc, ncsim, sim, prefix = 256, 256, 256, 'lowres/%d-9100-fixed'%256, 'lowres'
bs, nc, ncsim, sim, prefix = 256, 256, 2560, 'highres/%d-9100-fixed'%2560, 'highres'

# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
wsize = comm.size


if __name__=="__main__":

    for aa in aafiles[:]:


        suff = 'm1_00p3mh-alpha-0p8-subvol'
        halos = BigFileCatalog(scratch + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        comm = halos.comm
        mp = halos.attrs['MassTable'][1]*1e10
        cen = BigFileCatalog(myscratch + sim+ '/fastpm_%0.4f/cencat-%s/'%(aa, suff))
        sat = BigFileCatalog(myscratch + sim+ '/fastpm_%0.4f/satcat-%s/'%(aa, suff))

##
        hmass = halos['Length'].compute() * mp
        cmass = cen["Mass"].compute()
        chmass = cen["HaloMass"].compute()
        smass = sat["Mass"].compute()
        hpos, cpos, spos = halos['Position'].compute(), cen['Position'].compute(), sat['Position'].compute()
        chid, shid = cen['GlobalID'].compute(), sat['GlobalID'].compute()
        cnsat = cen['Nsat'].compute()

        da = DistributedArray(shid, comm)
        N = da.bincount(shared_edges=False)

        
        print('rank, shid, N : ', rank, shid[:10], N.local[:10])
        print('rank, chid, csat, N : ', rank, chid[:10], cnsat[:10], N.local[:10])
        print('rank, cen.csize, N.cshape : ', rank, cen.csize, N.cshape)
        print('rank, cen.size, Nlocal.size : ', rank, cen.size, N.local.size)
    
        print(cen.csize - N.cshape)
        zerosize = (cen.csize - N.cshape[0])
        #start = (zerosize *rank // wsize)
        #end =  (zerosize *(rank+1) // wsize)
        #zeros = DistributedArray(np.zeros(end-start), comm=comm)
        print(zerosize, N.local.dtype)
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=N.local.dtype, comm=comm)
        zeros.local[...] = 0
        print(rank, cen.csize, N.cshape, zeros.cshape, N.cshape+ zeros.cshape)
        N2 = DistributedArray.concat(N, zeros, localsize=cnsat.size)
        
        print('rank, shid, N : ', rank, shid[:10], N2.local[:10])
        print('rank, chid, csat, N : ', rank, chid[:10], cnsat[:10], N2.local[:10])
        print('rank, cen.csize, N.cshape : ', rank, cen.csize, N2.cshape)
        print('rank, cen.size, Nlocal.size : ', rank, cen.size, N2.local.size)


        print(rank, cnsat - N2.local)
        print(np.allclose(cnsat - N2.local, 0))
        
        #Do for mass

        M = da.bincount(smass, shared_edges=False)
        zerosize = (cen.csize - N.cshape[0])
        start = (zerosize *rank // wsize)
        end =  (zerosize *(rank+1) // wsize)
        zeros = DistributedArray(np.zeros(end-start), comm=comm)
        Msat = DistributedArray.concat(M, zeros, localsize=cnsat.size)
        
        ratio = (Msat.local + cmass)/chmass
        print(rank, ratio.min(), ratio.max())
        print(np.allclose(ratio, 1))
