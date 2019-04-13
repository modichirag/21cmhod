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
parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-s', '--size', help='for small or big box', default='big')
parser.add_argument('-a', '--amp', help='amplitude for up/down sigma 8', default=None)
args = parser.parse_args()
if args.model == None:
    print('Specify a model name')
    sys.exit()
#print(args, args.model)

model = args.model #'ModelD'
boxsize = args.size
amp = args.amp
#
#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist = alist[-1:]

#Parameters, box size, number of mesh cells, simulation, ...
if boxsize == 'small':
    bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
elif boxsize == 'big':
    bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'
else:
    print('Box size not understood, should be "big" or "small"')
    sys.exit()

if amp is not None:
    if amp == 'up' or amp == 'dn':
        sim = sim + '-%s'%amp
    else:
        print('Amplitude should be "up" or "dn". Given : ', amp)
        sys.exit()
        


#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
HImodel = modeldict[model] #HImodels.ModelB
modelname = model #'galaxies'
mode = modedict[model]
ofolder = '../data/outputs/'

suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256
if amp is not None: suff = suff + "-%s"%amp

dpath = ofolder + suff + '/%s/'%modelname

def getomHI():

    bbs, omz = [], []
    for ia, aa in enumerate(alist):
        # Read the data from file.
        zz = 1/aa-1
        omHI = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa)).T
        omHI = (omHI[1]*omHI[2]).sum()/bs**3/27.754e10 *(1+zz)**3
        omHI /= np.sqrt( 0.3*aa**-3+0.7 )**2
        omz.append(omHI)    
    return np.array(omz)



def getb1():
    
    bbs = []
    for ia, aa in enumerate(alist):
        bias = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:10, 1].mean()
        bbs.append(bias)
    return np.array(bbs)

    
def getsn():
    
    sns = []
    for ia, aa in enumerate(alist):
        ff = open(dpath + 'HI_pks_1d_{:06.4f}.txt'.format(aa))
        tmp = ff.readline()
        sn = float(tmp.split('=')[1].split('.\n')[0])
        sns.append(sn)
    return np.array(sns)
    

def getfsat():
    
    fracs = []
    for ia, aa in enumerate(alist):
        dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
        dist = dist[dist[:,1] !=0]
        total = (dist[:, 1]*dist[:, 2]).sum()
        sats = (dist[:, 1]*dist[:, 4]).sum()
        fracs.append(sats/total)
    return np.array(fracs)



if __name__=="__main__":

    alist = np.array(alist)
    zlist = 1/alist - 1
    omHI = getomHI()
    b1 = getb1()
    sn = getsn()
    fsat = getfsat()
    
    ff = open(dpath + 'summaryHI.txt', 'w')
    ff.write("{: >6} {:>12s} {:>6s} {:>12s} {:>6s}\n".format("z", "OmHI", "b1", "SN", "fsat"))
    for i in range(zlist.size):
        ff.write("{:6.2f} {:12.4e} {:6.2f} {:12.4e} {:6.2f}\n".format(zlist[i], omHI[i], b1[i], sn[i], fsat[i]))
    ff.close()
        
    
