import numpy as np
from time import time
import re
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog, FFTPower
#
#
#Global, fixed things
scratch1 = '/global/cscratch1/sd/yfeng1/m3127/'
scratch2 = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc = 256, 512
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
wsize = pm.comm.size
if rank == 0: print('World size = ', wsize)

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
    
    

if __name__=="__main__":
    if rank == 0: print('Starting')

    if bs == 1024:
        censuff = '-16node'
        satsuff='-m1_5p0min-alpha_0p8-16node'
    elif bs == 256:
        censuff = ''
        satsuff='-m1_5p0min-alpha_0p8'

    pks = []
    for aa in alist[:2]:

        if rank ==0 : print('For z = %0.2f'%(1/aa-1))

        #cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat'%aa+censuff)
        cencat = BigFileCatalog(scratch1+sim+'/fastpm_%0.4f/LL-0.200'%aa)
        mp = cencat.attrs['M0'][0]*1e10
        cencat['Mass'] = cencat['Length'] * mp
        #cencat['HImass'] = HI_hod(cencat['Mass'],aa)

        h1mesh = pm.paint(cencat['Position'], mass=cencat['Mass'])
        #h1mesh = pm.paint(cencat['Position'])

        print('Rank, mesh.cmean() : ', rank, h1mesh.cmean())
        h1mesh /= h1mesh.cmean()

        #
        pkh1h1   = FFTPower(h1mesh,mode='1d').power

        # Extract the quantities we want and write the file.
        kk   = pkh1h1['k']
        sn   = pkh1h1.attrs['shotnoise']
        pk   = np.abs(pkh1h1['power'])
        pks.append(pk)

    #if rank ==0 : 
    header = 'k, P(k, z) : %s'%alist[:2]
    tosave = np.concatenate((kk.reshape(-1, 1), np.array(pks).T), axis=-1)
    if rank == 0: print(tosave[:5])
    np.savetxt('../data/pkdebug2-%d.txt'%wsize, tosave, header=header)
