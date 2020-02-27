import numpy as np
import re, os, sys
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, MultipleSpeciesCatalog, FFTPower, FieldMesh
from nbodykit     import setup_logging
from mpi4py       import MPI

import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='for small or big box', default='small')
parser.add_argument('-a', '--amp', help='amplitude for up/down sigma 8', default=None)
parser.add_argument('-n', '--nmesh', help='amplitude for up/down sigma 8', default=-1)
parser.add_argument('-f', '--fixed', help='fixed sims', default=1)
args = parser.parse_args()

boxsize = args.size
amp = args.amp
#
#
#Global, fixed things
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/cyril/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist    = [0.2000,0.2222,0.2500,0.2857,0.3333]
#alist    = [0.2500,0.2857,0.3333]
#alist = alist[-1:]

#Parameters, box size, number of mesh cells, simulation, ...
if boxsize == 'small':
    bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'HV%d'%2560, 'highres'
elif boxsize == 'big':
    bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'HV%d'%10240, 'highres'
else:
    print('Box size not understood, should be "big" or "small"')
    sys.exit()

if args.fixed != 1: sim += '-R'
else: sim += '-F'
if args.nmesh == -1: pass
else: nc = int(args.nmesh)

if amp is not None:
    if amp == 'up' or amp == 'dn':
        sim = sim + '-%s'%amp
    else:
        print('Amplitude should be "up" or "dn". Given : ', amp)
        sys.exit()
        

# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args)

#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 



def read_conversions(db):
    """Read the conversion factors we need and check we have the right time."""
    mpart,Lbox,rsdfac,acheck = None,None,None,None
    with open(db+"Header/attr-v2","r") as ff:
        for line in ff.readlines():
            mm = re.search("MassTable.*\#HUMANE\s+\[\s*0\s+(\d*\.\d*)\s*0+\s+0\s+0\s+0\s+\]",line)
            if mm != None:
                mpart = float(mm.group(1)) * 1e10
            mm = re.search("BoxSize.*\#HUMANE\s+\[\s*(\d+)\s*\]",line)
            if mm != None:
                Lbox = float(mm.group(1))
            mm = re.search("RSDFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                rsdfac = float(mm.group(1))
            mm = re.search("ScalingFactor.*\#HUMANE\s+\[\s*(\d*\.\d*)\s*\]",line)
            if mm != None:
                acheck = float(mm.group(1))
    if (mpart is None)|(Lbox is None)|(rsdfac is None)|(acheck is None):
        print(mpart,Lbox,rsdfac,acheck)
        raise RuntimeError("Unable to get conversions from attr-v2.")
    if np.abs(acheck-aa)>1e-4:
        raise RuntimeError("Read a={:f}, expecting {:f}.".format(acheck,aa))
    return(rsdfac)
    #




    
    

if __name__=="__main__":
    if rank==0: print('Starting')
    suff='-m1_00p3mh-alpha-0p8-subvol'

    for aa in alist:
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
        halocat = BigFileCatalog(scratchcm + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10##
        if rank == 0: print('Mass of the particle : %0.2e'%mp)

        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/LL-0.200'%aa)
        cencat['Mass'] = cencat['Length'] * mp
        satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satellites'%aa)
        rsdfac = read_conversions(scratchcm + sim+'/fastpm_%0.4f/'%aa)
        #

        modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
        modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
        for model in modeldict:
            HImodel = modeldict[model]
            modelname = model
            mode = modedict[model]

            HImodelz = HImodel(aa)
            los = [0,0,1]
            halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
            halocat['RSDpos'], cencat['RSDpos'], satcat['RSDpos'] = HImodelz.assignrsd(rsdfac, halocat, cencat, satcat, los=los)

            if rank == 0: print('Creating HI mesh in redshift space')
            h1mesh = HImodelz.createmesh(bs, nc, halocat, cencat, satcat, mode=mode, position='RSDpos', weight='HImass', tofield=True)           
            FieldMesh(h1mesh).save(scratchcm + sim+'/fastpm_%0.4f/HImeshz-N%04d/'%(aa, nc), dataset=modelname, mode='real')

            if rank == 0: print('Creating HI mesh in real space for bias')
            h1mesh = HImodelz.createmesh(bs, nc, halocat, cencat, satcat, mode=mode, position='Position', weight='HImass', tofield=True)
            FieldMesh(h1mesh).save(scratchcm + sim+'/fastpm_%0.4f/HImesh-N%04d/'%(aa, nc), dataset=modelname, mode='real')

            if rank == 0: print('Saved for %s'%modelname)
