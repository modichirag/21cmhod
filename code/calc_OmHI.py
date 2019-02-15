#!/usr/bin/env python3
#
# Using the halo catalogs from FastPM, and a fiducial MHI-Mhalo
# relation, compute OmegaHI(z).
#
import numpy as np
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog

import HImodels


#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]


bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'

# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model & configuration to use
HImodel = HImodels.ModelA
modelname = 'ModelA'
mode = 'halos'
#ofolder = '../data/outputs/'




class Cosmology:
    # A simple class primarily to compute rho_crit.
    def __init__(self):
        self.omm = 0.309167
        self.omx = 0.690833
        self.hub = 0.677
    def rhoCritCom(self,zz):
        """Returns the critical density (in Msun/h and Mpc/h units) at
           redshift zz, scaled by comoving volume."""
        rho = 2.7754e11*( self.omm*(1+zz)**3+self.omx )/(1+zz)**3
        return(rho)
        #



def HI_hod(mhalo,aa):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = (1+2*zz)/(2+2*zz)
    mcut= 1e9*( 1.8 + 15*(3*aa)**8 )
    norm= 2e9*np.exp(-1.8*zp1+0.05*zp1**2)
    norm= 3e5*(1+(3.5/zz)**6)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #


def calc_OmHI(aa, suff):
    """Sums over the halos to compute OmHI."""
    # Work out the base path.
    
    halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10##
    halocat['Mass'] = halocat['Length'].compute() * mp
    cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat'%aa+suff)
    satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat'%aa+suff)

    HImodelz = HImodel(aa)
    halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)

    if mode == 'halos': catalogs = [halocat]
    elif mode == 'galaxies': catalogs = [cencat, satcat]
    elif mode == 'all': catalogs = [halocat, cencat, satcat]

    
    rankweight = sum([cat['HImass'].sum().compute() for cat in catalogs])
    mHI = comm.allreduce(rankweight)

    rankweight = sum([(cat['HImass']**2).sum().compute() for cat in catalogs])
    mHI2 = comm.allreduce(rankweight)

    # Convert to OmegaHI.
    nbar   = mHI**2/mHI2/bs**3
    rhoHI  = mHI/bs**3
    cc     = Cosmology()
    OmHI   = rhoHI/cc.rhoCritCom(1/aa-1)
    # For now just print it.
    if rank == 0: print("{:6.2f} {:12.4e} {:12.4e}".format(1/aa-1,OmHI,nbar))
    #





if __name__=="__main__":

    suff='-m1_00p3mh-alpha-0p8-subvol'

    for aa in alist:
        calc_OmHI(aa, suff)
    #






