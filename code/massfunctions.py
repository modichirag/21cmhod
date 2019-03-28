import numpy as np
import re, os, sys
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI

import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')
sys.path.append('./utils')
from uvbg import moster, mbh, lq, qlf, mbhparams

#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='for small or big box', default='small')
args = parser.parse_args()

boxsize = args.size


#
#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
if boxsize == 'small':
    bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
elif boxsize == 'big':
    bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'
else:
    print('Box size not understood, should be "big" or "small"')
    sys.exit()


modelname = 'ModelA'
# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model & configuration to use
ofolder = '../data/outputs/'




def stellarmassfunc(aa, halocat, cencat, satcat, outfolder, mbins=None):
    '''Compute the fraction of HI in halos, centrals, satellites'''

    zz = 1/aa-1
    h = cosmodef['h']
    if rank==0: print('Calculating distribution')

    hmass = cencat['Mass'].compute()
    cmass = cencat['Mass'].compute()
    smass = satcat['Mass'].compute()
    allmass = np.concatenate((cmass, smass))
    stmass = moster(allmass, zz)/h
    sthmass = moster(hmass, zz)/h

    #num, binedge = np.histogram(np.log10(stellarmass), mbins)
    #wts, _ = np.histogram(stellarmass, 10**mbins, weights=stellarmass)      
    #xx = (wts/num)
    #yy = (num/0.2/bsh**3/np.log(10))

    htotal, hsize, hhtotal, hhsize = [], [], [], []
    for im in range(mbins.size-1):
        mask = (stmass >= mbins[im]) & (stmass < mbins[im+1])
        rankweight = (stmass*mask).sum()
        htotal.append(comm.allreduce(rankweight))
        rankweight = (mask).sum()
        hsize.append(comm.allreduce(rankweight))

        mask = (sthmass >= mbins[im]) & (sthmass < mbins[im+1])
        rankweight = (sthmass*mask).sum()
        hhtotal.append(comm.allreduce(rankweight))
        rankweight = (mask).sum()
        hhsize.append(comm.allreduce(rankweight))
        
    
   #
    if rank==0:
        tosave = np.zeros((len(hsize), 6))
        tosave[:, 0] = np.log10(mbins[:-1])
        tosave[:, 1] = np.log10(mbins[1:])
        tosave[:, 3] = hsize
        tosave[:, 2] = htotal / (tosave[:, 3])
        tosave[:, 5] = hhsize
        tosave[:, 4] = hhtotal / (tosave[:, 5])
        header = 'Bin-low, Bin-High, Stellar Mass, Number, Stellar Mass Halo, Number Halo \nbsh={:.3f}, log10binwidth={:.2f}'.format(bs/h, np.diff(np.log10(mbins))[0])
        np.savetxt(outfolder + "smf_{:6.4f}.txt".format(aa), tosave, fmt='%0.6e', header=header)




def halomassfunc(aa, halocat, cencat, satcat, outfolder, mbins=None):
    '''Compute the fraction of HI in halos, centrals, satellites'''

    zz = 1/aa-1
    h = cosmodef['h']
    if rank==0: print('Calculating distribution')

    hmass = halocat['Mass'].compute()

    htotal, hsize, h1total = [], [], []
    for im in range(mbins.size-1):
        mask = (hmass >= mbins[im]) & (hmass < mbins[im+1])
        rankweight = (hmass*mask).sum()
        htotal.append(comm.allreduce(rankweight))
        rankweight = (mask).sum()
        hsize.append(comm.allreduce(rankweight))
        
    
   #
    if rank==0:
        tosave = np.zeros((len(hsize), 4))
        tosave[:, 0] = np.log10(mbins[:-1])
        tosave[:, 1] = np.log10(mbins[1:])
        tosave[:, 3] = hsize
        tosave[:, 2] = htotal / (tosave[:, 3])
        header = 'Bin-low, Bin-High, Halo Mass, Number \nbs={:.3f}, log10binwidth={:.2f}'.format(bs, np.diff(np.log10(mbins))[0])
        np.savetxt(outfolder + "hmf_{:6.4f}.txt".format(aa), tosave, fmt='%0.6e', header=header)





def qlumfunc(aa, halocat, cencat, satcat, outfolder, mbins=None, switchon=0.01, eta=0.1, galscatter=0.3, bhscatter=0.3, lumscatter=0.3):
    '''Compute the fraction of HI in halos, centrals, satellites'''

    zz = 1/aa-1
    h = cosmodef['h']
    if rank==0: print('Calculating QLF')

    cmass = cencat['Mass'].compute()
    smass = satcat['Mass'].compute()

    censize, satsize = cencat['Mass'].size, satcat['Mass'].size
    cenid = np.random.choice(np.arange(censize), size = int(censize*switchon))
    satid = np.random.choice(np.arange(satsize), size = int(satsize*switchon))

    alpha, beta = mbhparams['%0.1f'%zz][1:]
    if rank == 0: print('Parameters are ', alpha, beta)
    cencat['blackhole'] = mbh(moster(cencat['Mass'].compute(), z=zz, scatter=galscatter), alpha, beta, scatter=bhscatter)
    satcat['blackhole'] = mbh(moster(satcat['Mass'].compute(), z=zz, scatter=galscatter), alpha, beta, scatter=bhscatter)
    cencat['luminosity'] = lq(cencat['blackhole'], fon=switchon, eta=eta, scatter=lumscatter)
    satcat['luminosity'] = lq(satcat['blackhole'], fon=switchon, eta=eta, scatter=lumscatter)

    alllum = np.concatenate((cencat['luminosity'], satcat['luminosity']))
    alllum = alllum[alllum > 0]
    lsun = 3.28e26
    alllum *= lsun

    print(alllum)
    mag14 = 72.5+0.29- 2.5*np.log10(alllum)
    print(mag14)

    htotal, hsize = [], []
    for im in range(mbins.size-1):
        mask = (mag14 >= mbins[im]) & (mag14 < mbins[im+1])
        rankweight = (mag14*mask).sum()
        htotal.append(comm.allreduce(rankweight))
        rankweight = (mask).sum()
        hsize.append(comm.allreduce(rankweight))
        
    
   #
    if rank==0:
        tosave = np.zeros((len(hsize), 4))
        tosave[:, 0] = mbins[:-1]
        tosave[:, 1] = mbins[1:]
        tosave[:, 3] = hsize
        tosave[:, 2] = htotal / (tosave[:, 3])

        header = 'Bin Low, Bin high, Magnitude, Number\nbs={:.3f}, binwidth={:.2f}'.format(bs, np.diff(mbins)[0])
        np.savetxt(outfolder + "qlf_{:6.4f}.txt".format(aa), tosave, fmt='%0.6e', header=header)
    
    

if __name__=="__main__":
    if rank==0: print('Starting')
    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:]
    if bs == 1024: outfolder = outfolder + "-big"
    outfolder += "/%s/"%modelname
    if rank == 0: print(outfolder)
    #outfolder = ofolder + suff[1:] + "/%s/"%modelname
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

        #HImodelz = HImodel(aa)
        #halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
        #cencat['HIsat'] = HImodelz.getinsat(satcat['HImass'].compute(), satcat['GlobalID'].compute(), 
        #                                    cencat.csize, cencat['Mass'].size, cencat.comm).local
       

        mbins = 10**np.arange(8, 12, 0.2)
        stellarmassfunc(aa, halocat, cencat, satcat, outfolder, mbins=mbins)

        mbins = 10**np.arange(9, 14, 0.2)
        halomassfunc(aa, halocat, cencat, satcat, outfolder, mbins=mbins)

        magbins = np.linspace(-28, -5, 50)
        try:
            qlumfunc(aa, halocat, cencat, satcat, outfolder, mbins=magbins)
        except: continue

