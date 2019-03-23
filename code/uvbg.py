import numpy as np
import re, os, sys
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI
import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Set random seed
np.random.seed(100)


#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]

#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 


mbhparams = {'5.0':[1e-2, -4, 0.75], '6.0':[1e-2, -3.5, 1]}


def mfpathz(z):
    return 37*((1+z)/5)**-5.4 * (1+z) #multiply 1+z to convert to comoving


def moster(Mhalo,z,h=0.6776, scatter=None):
    """
    """
    Minf = Mhalo/h
    zzp1  = z/(1+z)
    M1    = 10.0**(11.590+1.195*zzp1)
    mM    = 0.0351 - 0.0247*zzp1
    beta  = 1.376  - 0.826*zzp1
    gamma = 0.608  + 0.329*zzp1
    Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
    Mstar*= Minf
    if scatter is not None: 
        Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
    return Mstar*h
    #
                                                                                                      

def qlf(M, zz, alpha=-2.03, beta=-4, Ms=-27.21, lphi6 = -8.94):
    phis = 10**(lphi6 -0.47*(zz-6))
    f1 = 10**(0.4*(alpha+1)*(M - Ms))
    f2 = 10**(0.4*(beta+1)*(M - Ms))
    return phis/(f1+f2)


def mbh(mg, alpha=-3.5, beta=1, scatter=False, h=0.677):
    m = mg/h
    mb = 1e10 * 10**alpha * (m/1e10)**beta
    if scatter: mb = 10**(np.log10(mb) + np.random.normal(scale=scatter, size=mb.size))
    return mb*h


def lq(mb, fon=0.01, eta=0.1, scatter=False, h=0.677):
    m = mb/h
    lsun = 3.28e26*np.ones_like(m)
    lsun = 1.*np.ones_like(m)
    if scatter: eta = np.random.lognormal(eta, scatter, m.size)
    indices = np.random.choice(np.arange(m.size).astype(int), replace=False, size=int(m.size*(1-fon)))
    lsun[indices] = 0 
    return 3.3e4*eta*m *lsun


def modulateHI(pos, mfid, uvmesh, layout, alpha, kappa=None):
    if kappa is None: kappa = (1-2*alpha)/alpha
    uvmean = uvmesh.cmean()
    uvx = uvmesh.readout(pos, layout=layout)
    index = (3-alpha)/alpha/kappa
    mnew = mfid * (uvx / uvmean)**index
    return mnew


def setupuvmesh(zz, suff, sim, profile, pm, model='ModelA', switchon=0.01, eta=0.1, galscatter=0.3, bhscatter=0.3, lumscatter=0.3):
    
    rank = pm.comm.rank
    aa = 1/(1+zz)
    if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
    halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    halocat['Mass'] = halocat['Length'].compute() * mp
    cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat-'%aa+suff)
    satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat-'%aa+suff)
    clayout = pm.decompose(cencat['Position'])
    slayout = pm.decompose(satcat['Position'])
    #

    HImodel = modeldict[model] #HImodels.ModelB
    HImodelz = HImodel(aa)
    halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
    cmeshfid = pm.paint(cencat['Position'], mass=cencat['HImass'], layout=clayout)
    smeshfid = pm.paint(satcat['Position'], mass=satcat['HImass'], layout=slayout)
    h1meshfid = cmeshfid + smeshfid
    
    censize, satsize = cencat['Mass'].size, satcat['Mass'].size
    cenid = np.random.choice(np.arange(censize), size = int(censize*switchon))
    satid = np.random.choice(np.arange(satsize), size = int(satsize*switchon))

    alpha, beta = mbhparams['%0.1f'%zz][1:]
    if rank == 0: print('Parameters are ', alpha, beta)
    cencat['blackhole'] = mbh(moster(cencat['Mass'].compute(), z=zz, scatter=galscatter), alpha, beta, scatter=bhscatter)
    satcat['blackhole'] = mbh(moster(satcat['Mass'].compute(), z=zz, scatter=galscatter), alpha, beta, scatter=bhscatter)
    cencat['luminosity'] = lq(cencat['blackhole'], fon=switchon, eta=eta, scatter=lumscatter)
    satcat['luminosity'] = lq(satcat['blackhole'], fon=switchon, eta=eta, scatter=lumscatter)

    cmesh = pm.paint(cencat['Position'], mass=cencat['luminosity'], layout=clayout)
    smesh = pm.paint(satcat['Position'], mass=satcat['luminosity'], layout=slayout)
    lmesh = cmesh + smesh
    #if lumspectra : calc_bias(aa,lmesh/lmesh.cmean(), suff, fname='Lum')


    mfpath = mfpathz(zz)
    if rank == 0: print('At redshift {:.2f}, mean free path is {:.2f}'.format(zz, mfpath))
    meshc = lmesh.r2c()
    kmesh = sum(i**2 for i in meshc.x)**0.5
    wt = np.arctan(kmesh *mfpath)/kmesh/mfpath
    wt[kmesh == 0] = 1
    meshc*= wt
    uvmesh = meshc.c2r()
    #print(uvmesh.cmean())
    #if uvspectra: calc_bias(aa, uvmesh/uvmesh.cmean(), suff, fname='UVbg')

    #
    cencat['HIuvmass'] = modulateHI(cencat['Position'], cencat['HImass'], uvmesh, clayout, alpha=profile)
    satcat['HIuvmass'] = modulateHI(satcat['Position'], satcat['HImass'], uvmesh, slayout, alpha=profile)

    cmesh = pm.paint(cencat['Position'], mass=cencat['HIuvmass'], layout=clayout)
    smesh = pm.paint(satcat['Position'], mass=satcat['HIuvmass'], layout=slayout)
    h1mesh = cmesh + smesh
    #calc_bias(aa, h1mesh/h1mesh.cmean(), suff, fname='HI_UVbg_ap%dp%d'%((profile*10)//10, (profile*10)%10))

    cats = [cencat, satcat]
    meshes = [h1meshfid, h1mesh, lmesh, uvmesh]
    return cats, meshes



    
    
##
##if __name__=="__main__":
##    if rank==0: print('Starting')
##    suff='m1_00p3mh-alpha-0p8-subvol'
##    outfolder = ofolder + suff
##    if bs == 1024: outfolder = outfolder + "-big"
##    outfolder += "/%s/"%modelname
##    if rank == 0: print(outfolder)
##    try: 
##        os.makedirs(outfolder)
##    except : pass
##
##    #for aa in alist:
##    for zz in [6.0, 5.0]:
##        aa = 1/(1+zz)
##
##        cats, meshes = setupuvmesh(zz, suff=suff, sim=sim, profile=profile)
##        cencat, satcat = cats
##        h1meshfid, h1mesh, lmesh, uvmesh = meshes
##
##
##        if lumspectra : calc_bias(aa,lmesh/lmesh.cmean(), outfolder, fname='Lum')
##
##        if uvspectra: calc_bias(aa, uvmesh/uvmesh.cmean(), outfolder, fname='UVbg')
##
##        fname = 'HI_UVbg_ap%dp%d'%((profile*10)//10, (profile*10)%10)
##        calc_bias(aa, h1mesh/h1mesh.cmean(), outfolder, fname=fname)
##
##        ratio = (cencat['HIuvmass']/cencat['HImass']).compute()
##        print(rank, 'Cen', '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())
##
##        ratio = (satcat['HIuvmass']/satcat['HImass']).compute()
##        print(rank, 'Sat', '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())
##
###        uvpreview = uvmesh.preview(Nmesh=128)
###        if rank == 0:
###            import matplotlib.pyplot as plt
###            from matplotlib.colors import LogNorm
###            print('save figure')
###            plt.figure()
###            plt.imshow(uvpreview.sum(axis=0), norm=LogNorm())
###            plt.colorbar()
###            plt.savefig('tmpfig%d.pdf'%zz)
###            print('Figure saved')
###            
