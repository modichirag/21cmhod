import numpy as np
import re, os, sys
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI

import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='for small or big box', default='small')
args = parser.parse_args()
if args.model == None:
    print('Specify a model name')
    sys.exit()
#print(args, args.model)

model = args.model #'ModelD'
boxsize = args.size

#
#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist = alist[5:]

#Parameters, box size, number of mesh cells, simulation, ...
if boxsize == 'small':
    bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
elif boxsize == 'big':
    bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'
else:
    print('Box size not understood, should be "big" or "small"')
    sys.exit()

#bs, nc, ncsim, sim, prefix = 256, 128, 256, 'lowres/%d-9100-fixed'%256, 'lowres'


# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
HImodel = modeldict[model] #HImodels.ModelB
modelname = model #'galaxies'
mode = modedict[model]
ofolder = '../data/outputs/'


params = {'5.0':[1e-2, -4, 0.75], '6.0':[1e-2, -3.5, 1]}


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
    if scatter: eta = np.random.lognormal(eta, scatter, m.size)
    indices = np.random.choice(np.arange(m.size).astype(int), replace=False, size=int(m.size*(1-fon)))
    lsun[indices] = 0 
    return 3.3e4*eta*m *lsun


def modulateHI(pos, mfid, uvmesh, layout, alpha=2.9, kappa=None):
    if kappa is None: kappa = (1-2*alpha)/alpha
    uvmean = uvmesh.cmean()
    uvx = uvmesh.readout(pos, layout=layout)
    index = (3-alpha)/alpha/kappa
    mnew = mfid * (uvx / uvmean)**index
    return mnew



    
def calc_bias(aa,h1mesh,suff, fname):
    '''Compute the bias(es) for the HI'''

    if rank==0: print('Calculating bias')
    if rank==0:
        print("Processing a={:.4f}...".format(aa))
        print('Reading DM mesh...')
    if ncsim == 10240:
        dm    = BigFileMesh(scratchyf+sim+'/fastpm_%0.4f/'%aa+\
                            '/1-mesh/N%04d'%nc,'').paint()
    else:
        dm    = BigFileMesh(project+sim+'/fastpm_%0.4f/'%aa+\
                        '/dmesh_N%04d/1/'%nc,'').paint()
    dm   /= dm.cmean()
    if rank==0: print('Computing DM P(k)...')
    pkmm  = FFTPower(dm,mode='1d').power
    k,pkmm= pkmm['k'],pkmm['power']  # Ignore shotnoise.
    if rank==0: print('Done.')
    #

    pkh1h1 = FFTPower(h1mesh,mode='1d').power
    kk = pkh1h1.coords['k']

    pkh1h1 = pkh1h1['power']-pkh1h1.attrs['shotnoise']
    pkh1mm = FFTPower(h1mesh,second=dm,mode='1d').power['power']
    if rank==0: print('Done.')
    # Compute the biases.
    b1x = np.abs(pkh1mm/(pkmm+1e-10))
    b1a = np.abs(pkh1h1/(pkmm+1e-10))**0.5
    if rank==0: print("Finishing processing a={:.4f}.".format(aa))

    #
    if rank==0:
        fout = open(outfolder + "{}_bias_{:6.4f}.txt".format(fname, aa),"w")
        fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                   format("k","b1_x","b1_a","Pkmm"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                       format(kk[i],b1x[i],b1a[i],pkmm[i].real))
        fout.close()

    
    

if __name__=="__main__":
    if rank==0: print('Starting')
    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:]
    if bs == 1024: outfolder = outfolder + "-big"
    outfolder += "/%s/"%modelname
    if rank == 0: print(outfolder)
    try: 
        os.makedirs(outfolder)
    except : pass

    #for aa in alist:
    for zz in [6.0, 5.0]:
        aa = 1/(1+zz)
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))

        halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10
        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat'%aa+suff)
        satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat'%aa+suff)
        #

        HImodelz = HImodel(aa)
        halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)

        switchon = 0.01
        censize, satsize = cencat['Mass'].size, satcat['Mass'].size
        cenid = np.random.choice(np.arange(censize), size = int(censize*switchon))
        satid = np.random.choice(np.arange(satsize), size = int(satsize*switchon))

        alpha, beta = params['%0.1f'%zz][1:]
        if rank == 0: print('Parameters are ', alpha, beta)
        cencat['blackhole'] = mbh(moster(cencat['Mass'].compute(), z=zz, scatter=0.3), alpha, beta, scatter=0.3)
        satcat['blackhole'] = mbh(moster(satcat['Mass'].compute(), z=zz, scatter=0.3), alpha, beta, scatter=0.3)
        cencat['luminosity'] = lq(cencat['blackhole'], fon=switchon, eta=0.1, scatter=0.3)
        satcat['luminosity'] = lq(satcat['blackhole'], fon=switchon, eta=0.1, scatter=0.3)

        clayout = pm.decompose(cencat['Position'])
        slayout = pm.decompose(satcat['Position'])
        cmesh = pm.paint(cencat['Position'], mass=cencat['luminosity'], layout=clayout)
        smesh = pm.paint(satcat['Position'], mass=satcat['luminosity'], layout=slayout)
        lmesh = cmesh + smesh


        mfpath = mfpathz(zz)
        meshc = lmesh.r2c()
        kmesh = sum(i**2 for i in meshc.x)**0.5
        wt = np.arctan(kmesh *mfpath)/kmesh/mfpath
        wt[kmesh == 0] = 1
        meshc*= wt
        uvmesh = meshc.c2r()
        #print(uvmesh.cmean())
        calc_bias(aa, uvmesh/uvmesh.cmean(), suff, fname='UVbg')

        #
        cencat['HIuvmass'] = modulateHI(cencat['Position'], cencat['HImass'], uvmesh, clayout)
        satcat['HIuvmass'] = modulateHI(satcat['Position'], satcat['HImass'], uvmesh, slayout)
        
        cmesh = pm.paint(cencat['Position'], mass=cencat['HIuvmass'], layout=clayout)
        smesh = pm.paint(satcat['Position'], mass=satcat['HIuvmass'], layout=slayout)
        h1mesh = cmesh + smesh
        calc_bias(aa, h1mesh/h1mesh.cmean(), suff, fname='HI_UVbg')

        ratio = (cencat['HIuvmass']/cencat['HImass']).compute()
        print(rank, ratio[:10])
        print(rank, '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())

        ratio = (satcat['HIuvmass']/satcat['HImass']).compute()
        print(rank, ratio[:10])
        print(rank, '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())

#        uvpreview = uvmesh.preview(Nmesh=128)
#        if rank == 0:
#            import matplotlib.pyplot as plt
#            from matplotlib.colors import LogNorm
#            print('save figure')
#            plt.figure()
#            plt.imshow(uvpreview.sum(axis=0), norm=LogNorm())
#            plt.colorbar()
#            plt.savefig('tmpfig%d.pdf'%zz)
#            print('Figure saved')
#            
