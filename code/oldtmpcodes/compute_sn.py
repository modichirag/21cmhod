import numpy as np
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog,\
                         BigFileMesh, FFTPower
#
#Global, fixed things
scratch1 = '/global/cscratch1/sd/yfeng1/m3127/'
scratch2 = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist   = [0.1429,0.1667,0.2000,0.2222,0.2500,0.2857,0.3333]
alist   = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist   = [0.2857,0.3333]

#Parameters, box size, number of mesh cells, simulation, ...
bs,nc,ncsim = 1024, 1024, 10240
bs,nc,ncsim = 256, 256, 2560
sim,prefix  = 'highres/%d-9100-fixed'%ncsim, 'highres'

# It's useful to have my rank for printing...
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank


def HI_hod(mhalo,aa,mcut=2e9):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = 1.0
    alp = (1+2*zz)/(2+2*zz)
    norm= 3e5*(1+(3.5/zz)**6)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #


    
    

if __name__=="__main__":
    encsuff = '-16node'
    satsuff='-m1_5p0min-alpha_0p8-16node'
    censuff = ''
    satsuff='-m1_5p0min-alpha_0p8'
    
    sn = []
    for aa in alist[:1]:
        zz   = 1.0/aa-1.0
        mcut = 2e10*aa
        mcut = 1e10*np.exp(-(zz-2.0))
        mcut = 1.5e10*(3*aa)**3
        mcut = 3e9
        mcut = 1e9*( 1.8 + 15*(3*aa)**8 )
    
        #if rank==0: print('Reading central/satellite catalogs...')
        #cencat = BigFileCatalog(scratch1+sim+'/fastpm_%0.4f/LL-0.200'%aa)
        cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat'%aa+censuff)
        satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+satsuff)
       # if rank==0: print('Catalogs read.')
        #
        #if rank==0: print('Computing HI masses...')

        cencat['HImass'] = HI_hod(cencat['Mass'],aa,mcut)   
        satcat['HImass'] = HI_hod(satcat['Mass'],aa,mcut)   
        totHImass        = cencat['HImass'].sum().compute() +\
                           satcat['HImass'].sum().compute()
        wt_eff =  ((cencat['HImass']**2).sum() +( satcat['HImass']**2).sum()).compute()/totHImass**2    
        sn.append(bs**3 /wt_eff)
        print('Rank, totHImass :' , rank, '%0.3e'%totHImass)
##        
##    if rank==0:
##        fout = open("../data/L{:04d}/HI_shotnoise.txt".format(bs),"w")
##        fout.write("# Mcut={:12.4e}Msun/h.\n".format(mcut))
##        fout.write("# {:>10s} {:>10s}n".\
##                   format("a","sn"))
##        for ia, aa in enumerate(alist):
##            fout.write("{:10.4f} {:10.5e}\n".\
##                       format(aa, sn[ia]))
##        fout.close()
###
