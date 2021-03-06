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
#parser.add_argument('pathcm', help='path of input data from cm scratch')
parser.add_argument('-p', '--parameter',  help='parameter to change', default='alpha')
parser.add_argument('-m', '--model', help='model name to use', default='ModelC')
parser.add_argument('-s', '--size', help='for small or big box', default='small')
parser.add_argument('-a', '--amp', help='amplitude for up/down sigma 8', default=None)
parser.add_argument('-d', '--delta', help='change the parameter by', default=0.05, type=float)
args = parser.parse_args()
if args.parameter == None:
    print('Specify a parameter to vary')
    sys.exit()
#print(args, args.model)

model = args.model #'ModelD'
boxsize = args.size
amp = args.amp
#param = args.parameter
#delta = args.delta
#scratchcm = args.pathcm
#
#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist    = [0.1429,0.2000,0.3333]
#alist = [0.3333]

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
        

# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
if rank == 0: print(args)

#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
HImodel = modeldict[model] #HImodels.ModelB
modelname = model #'galaxies'
mode = modedict[model]
ofolder = '../data/outputs/'





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




def calc_pk1d(aa, h1mesh, outfolder):
    '''Compute the 1D redshift-space P(k) for the HI'''

    if rank==0: print('Calculating pk1d')
    pkh1h1   = FFTPower(h1mesh,mode='1d',kmin=0.025,dk=0.0125).power
    # Extract the quantities we want and write the file.
    kk   = pkh1h1['k']
    sn   = pkh1h1.attrs['shotnoise']
    pk   = np.abs(pkh1h1['power'])
    if rank==0:
        fout = open(outfolder + "HI_pks_1d_{:6.4f}.txt".format(aa),"w")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_0_HI"))
        for i in range(kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]-sn))
        fout.close()
    #




def calc_pkmu(aa, h1mesh, outfolder, los=[0,0,1]):
    '''Compute the redshift-space P(k) for the HI in mu bins'''

    if rank==0: print('Calculating pkmu')
    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=4,los=los).power
    # Extract what we want.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    pk = pkh1h1['power']
    # Write the results to a file.
    if rank==0:
        fout = open(outfolder + "HI_pks_mu_{:06.4f}.txt".format(aa),"w")
        fout.write("# Redshift space power spectrum in mu bins.\n")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        ss = "# {:>8s}".format(r'k\mu')
        for i in range(pkh1h1.shape[1]):
            ss += " {:15.5f}".format(pkh1h1.coords['mu'][i])
        fout.write(ss+"\n")
        for i in range(1,pk.shape[0]):
            ss = "{:10.5f}".format(kk[i])
            for j in range(pk.shape[1]):
                ss += " {:15.5e}".format(np.abs(pk[i,j]-sn))
            fout.write(ss+"\n")
        fout.close()
    #




def calc_pkll(aa, h1mesh, outfolder, los=[0,0,1]):
    '''Compute the redshift-space P_ell(k) for the HI'''

    if rank==0: print('Calculating pkll')
    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=8,los=los,\
                      kmin=0.02,dk=0.02,poles=[0,2,4]).poles
    # Extract the quantities of interest.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    P0 = pkh1h1['power_0'].real - sn
    P2 = pkh1h1['power_2'].real
    P4 = pkh1h1['power_4'].real
    # Write the results to a file.
    if rank==0:
        fout = open(outfolder + "HI_pks_ll_{:06.4f}.txt".format(aa),"w")
        fout.write("# Redshift space power spectrum multipoles.\n")
        fout.write("# Subtracting SN={:15.5e} from monopole.\n".format(sn))
        fout.write("# {:>8s} {:>15s} {:>15s} {:>15s}\n".\
                   format("k","P0","P2","P4"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:15.5e} {:15.5e} {:15.5e}\n".\
                       format(kk[i],P0[i],P2[i],P4[i]))
        fout.close()
    #



def calc_bias(aa,h1mesh,suff):
    '''Compute the bias(es) for the HI'''

    if rank==0: print('Calculating bias')
    if rank==0:
        print("Processing a={:.4f}...".format(aa))
        print('Reading DM mesh...')
    if ncsim == 10240:
        dm    = BigFileMesh(scratchyf+sim+'/fastpm_%0.4f/'%aa+\
                            '/1-mesh/N%04d'%nc,'').paint()
    else:
        dm    = BigFileMesh(scratchcm+sim+'/fastpm_%0.4f/'%aa+\
                            '/1-mesh/', '1').paint()
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
        fout = open(outfolder + "HI_bias_{:6.4f}.txt".format(aa),"w")
        fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                   format("k","b1_x","b1_a","Pkmm"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                       format(kk[i],b1x[i],b1a[i],pkmm[i].real))
        fout.close()




def get_changeM0(aa, Model):

    zz = 1/aa - 1
    mod = Model(aa) 
    mcut = mod.mcut
    mod.derivate('mcut', 0.05)
    mcutup = mod.mcut
    mod = Model(aa) 
    mod.derivate('mcut', -0.05)
    mcutdn = mod.mcut
    
    mdiff = mcutup - mcutdn
    if boxsize == 'small':
        dpath = '../data/outputs/m1_00p3mh-alpha-0p8-subvol/ModelC/'
        dpathup = '../data/outputs/m1_00p3mh-alpha-0p8-subvol-up/ModelC/'
        dpathdn = '../data/outputs/m1_00p3mh-alpha-0p8-subvol-dn/ModelC/'
    else:
        dpath = '../data/outputs/m1_00p3mh-alpha-0p8-subvol-big/ModelC/'
        dpathup = '../data/outputs/m1_00p3mh-alpha-0p8-subvol-big-up/ModelC/'
        dpathdn = '../data/outputs/m1_00p3mh-alpha-0p8-subvol-big-dn/ModelC/'

    fid = np.loadtxt(dpath + 'HI_bias_{:06.4f}.txt'.format(aa)).T
    kk = fid[0]
    pkf = fid[2]**2*fid[3]
    bba = fid[2][1:6].mean()

    p1 = np.loadtxt(dpathup + 'HI_bias_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpathdn + 'HI_bias_{:06.4f}.txt'.format(aa)).T
    bbaup, bbadn = p1[2][1:6].mean(), p2[2][1:6].mean()
    bbxup, bbxdn = p1[1][1:6].mean(), p2[1][1:6].mean()
    if rank == 0: print('Bias fid, up, dn : ', bba, bbaup, bbadn)

    derivba, derivbx = 0, 0
    p1 = np.loadtxt(dpath + 'mcut_vp05/HI_bias_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpath + 'mcut_vm05/HI_bias_{:06.4f}.txt'.format(aa)).T
    
    bb1a, bb1x = p1[2][1:6].mean(), p1[1][1:6].mean()
    bb2a, bb2x = p2[2][1:6].mean(), p2[1][1:6].mean()
    
    dbba = (bb1a - bb2a)/mdiff
    dbbx = (bb1x - bb2x)/mdiff
    
    dmaup = (bba - bbaup)/dbba
    dmadn = (bba - bbadn)/dbba
    return dmaup, dmadn*0.8 #Fudge for lower, not sure why
    
    

if __name__=="__main__":
    if rank==0: print('Starting')
    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:] 
    if bs == 1024: outfolder = outfolder + "-big"
    if amp is not None: outfolder = outfolder + "-%s"%amp
    outfolder += "/%s/"%modelname
    outfolder += '/matchbias_mcut/'
    
    if rank == 0: print(outfolder)
    try: 
        os.makedirs(outfolder)
    except : pass

    for aa in alist:
        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
        halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10##
        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm+sim+'/fastpm_%0.4f/cencat'%aa+suff)
        satcat = BigFileCatalog(scratchcm+sim+'/fastpm_%0.4f/satcat'%aa+suff)
        rsdfac = read_conversions(scratchcm+sim+'/fastpm_%0.4f/'%aa)
        #
        #

        HImodelz = HImodel(aa)
        mcut = HImodelz.mcut
        dmup, dmdn = get_changeM0(aa, HImodel)
        if rank == 0: print ('Fractional change in mcut ', dmup/mcut, dmdn/mcut)
        if amp == 'up':
            HImodelznew = HImodel(aa)
            HImodelznew.mcut   += dmup
        else:
            HImodelznew = HImodel(aa)
            HImodelznew.mcut   += dmdn

        los = [0,0,1]
        halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelznew.assignHI(halocat, cencat, satcat)
        halocat['RSDpos'], cencat['RSDpos'], satcat['RSDpos'] = HImodelznew.assignrsd(rsdfac, halocat, cencat, satcat, los=los)

        if rank == 0: print('Creating HI mesh in redshift space')
        h1mesh = HImodelznew.createmesh(bs, nc, halocat, cencat, satcat, mode=mode, position='RSDpos', weight='HImass')

        calc_pk1d(aa, h1mesh, outfolder)
        calc_pkmu(aa, h1mesh, outfolder, los=los)
        calc_pkll(aa, h1mesh, outfolder, los=los)

        if rank == 0: print('Creating HI mesh in real space for bias')
        h1mesh = HImodelznew.createmesh(bs, nc, halocat, cencat, satcat, mode=mode, position='Position', weight='HImass')
        calc_bias(aa, h1mesh, outfolder)


