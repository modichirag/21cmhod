import numpy as np
import re, os
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI

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


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs,nc,ncsim, sim, prefic = 1024, 1024, 10240, 'highres/%d-9100-fixed'%ncsim, 'highres'


# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model & configuration to use
HImodel = HImodels.ModelA
modelname = 'ModelA'
ofolder = '../data/outputs/'




def distribution(aa, halocat, cencat, satcat, outfolder, mbins=None):
    '''Compute the fraction of HI in halos, centrals, satellites'''

    if rank==0: print('Calculating distribution')

    if mbins is None: mbins = np.logspace(9, 15, 100)
    hmass = halocat['Mass'].compute()


    htotal, hsize, h1total = [], [], []
    for im in range(mbins.size-1):
        mask = (hmass >= mbins[im]) & (hmass < mbins[im+1])
        rankweight = (hmass*mask).sum()
        htotal.append(comm.allreduce(rankweight))
        rankweight = (mask).sum()
        hsize.append(comm.allreduce(rankweight))
        
        h1bin = []
        for cat in [halocat['HImass'], cencat['HImass'], cencat['HIsat']]:
            rankweight = (cat.compute()*mask).sum()
            h1bin.append(comm.allreduce(rankweight))
        h1total.append(h1bin)

    
    #
    if rank==0:
        tosave = np.zeros((len(hsize), 5))
        tosave[:, 1] = hsize
        tosave[:, 0] = htotal / (tosave[:, 1] + 1)
        tosave[:, 2:] = h1total/ (tosave[:, 1] + 1)
        header = 'Halo Mass, Number Halos, HI halos, HI centrals, HI satellites'
        np.savetxt(outfolder + "HI_dist_{:6.4f}.txt".format(aa), tosave, fmt='%0.6e', header=header)
    
    

if __name__=="__main__":
    if rank==0: print('Starting')
    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:] + "/%s/"%modelname
    try: 
        os.makedirs(outfolder)
    except : pass

    for aa in alist:
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
        halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10##
        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat'%aa+suff)
        satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat'%aa+suff)
        #

        HImodelz = HImodel(aa)
        halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
        cencat['HIsat'] = HImodelz.getinsat(satcat['HImass'].compute(), satcat['GlobalID'].compute(), 
                                            cencat.csize, cencat['Mass'].size, cencat.comm).local
        

        distribution(aa, halocat, cencat, satcat, outfolder)

