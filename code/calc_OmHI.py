#!/usr/bin/env python3
#
# Using the halo catalogs from FastPM, and a fiducial MHI-Mhalo
# relation, compute OmegaHI(z).
#
import numpy as np
from nbodykit.lab import BigFileCatalog




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
    mcut= 2e10*aa
    norm= 2e9*np.exp(-1.8*zp1+0.05*zp1**2)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #







def calc_OmHI(aa):
    """Sums over the halos to compute OmHI."""
    # Work out the base path.
    Lbox=256.0
    suff='-m1_5p0min-alpha_0p9'
    db = "/project/projectdirs/m3127/H1mass/highres/2560-9100-fixed/"+\
         "fastpm_{:06.4f}/".format(aa)
    #
    cmass  = BigFileCatalog(db+'cencat')['Mass']
    smass  = BigFileCatalog(db+'satcat'+suff)['Mass']
    ch1mass= HI_hod(cmass,aa)   
    sh1mass= HI_hod(smass,aa)   
    mHI    = np.sum(ch1mass.compute())+np.sum(sh1mass.compute())
    # Convert to OmegaHI.
    rhoHI = mHI/Lbox**3
    cc    = Cosmology()
    OmHI  = rhoHI/cc.rhoCritCom(1/aa-1)
    # For now just print it.
    print("{:6.2f} {:12.4e}".format(1/aa-1,OmHI))
    #





if __name__=="__main__":
    for aa in [0.3333,0.2857,0.2500,0.2222,0.2000,0.1818,0.1667,0.1538,0.1429]:
        calc_OmHI(aa)
    #
