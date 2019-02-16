import numpy as np
from time import time
import re
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog, FFTPower
from nbodykit.transform import HaloRadius, HaloVelocityDispersion
from nbodykit.cosmology.cosmology import Cosmology
#
#
#Global, fixed things
scratch1 = '/global/cscratch1/sd/yfeng1/m3127/'
scratch2 = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
print(cosmo)
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist    = [0.2222,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc = 256, 512
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs,nc,ncsim = 1024, 1024, 10240
sim,prefix  = 'highres/%d-9100-fixed'%ncsim, 'highres'
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank


if __name__=="__main__":
    print('Starting')

##    satsuff='-mmin0p1_m1_5p0min-alpha_0p9'
##    satsuff='-m1_8p0min-alpha_0p9'
##    satsuff='-m1_5p0min-alpha_0p9'
##
    if bs == 1024:
        censuff = '-16node'
        satsuff='-m1_5p0min-alpha_0p8-16node'
    elif bs == 256:
        censuff = ''
        satsuff='-m1_5p0min-alpha_0p8'

    for aa in alist:
        print(aa)
        if rank ==0 : print('For z = %0.2f'%(1/aa-1))
        if rank ==0 : print('Read in central/satellite catalogs')

        start = time()
        cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat'%aa+censuff)
        satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+satsuff)
        cmass = np.concatenate((cencat['Mass'].compute()[:5], cencat['Mass'].compute()[-5:]))
        print(cmass)
        print('R : ', HaloRadius(cmass, cosmo, 1/aa-1).compute()/aa)
        print('Vdisp : ', HaloVelocityDispersion(cmass, cosmo, 1/aa-1).compute())


#        spos = satcat['Position'].compute()
#        cpos = cencat['Position'].compute()
#        cvel = cencat['Velocity'].compute()
#        print((cvel[:10]**2).sum(axis=-1)**0.5)
#        smask = satcat['HaloID'].compute() == 1000
#        cmask = cencat['HaloID'].compute() == 1000
#        dist = ((spos[smask] - cpos[cmask])**2).sum(axis=-1)**0.5
#        print(dist)
#        print(dist.min(), dist.max())
