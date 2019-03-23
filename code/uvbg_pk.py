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
from uvbg import setupuvmesh

#Set random seed
np.random.seed(100)


#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='for small or big box', default='small')
parser.add_argument('-p', '--profile', help='slope of profile', default=2.9, type=float)
args = parser.parse_args()
#print(args, args.model)

model = args.model #'ModelD'
boxsize = args.size
profile = args.profile

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



# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm
uvspectra = True
lumspectra = True

#Which model & configuration to use
modeldict = {'ModelA':HImodels.ModelA, 'ModelB':HImodels.ModelB, 'ModelC':HImodels.ModelC}
modedict = {'ModelA':'galaxies', 'ModelB':'galaxies', 'ModelC':'halos'} 
HImodel = modeldict[model] #HImodels.ModelB
modelname = model #'galaxies'
mode = modedict[model]
ofolder = '../data/outputs/'



    
def calc_bias(aa,h1mesh, outfolder, fname):
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
    suff='m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff
    if bs == 1024: outfolder = outfolder + "-big"
    outfolder += "/%s/"%modelname
    if rank == 0: print(outfolder)
    try: 
        os.makedirs(outfolder)
    except : pass

    #for aa in alist:
    for zz in [6.0, 5.0]:
        aa = 1/(1+zz)

        cats, meshes = setupuvmesh(zz, suff=suff, sim=sim, profile=profile, pm=pm)
        cencat, satcat = cats
        h1meshfid, h1mesh, lmesh, uvmesh = meshes


        if lumspectra : calc_bias(aa,lmesh/lmesh.cmean(), outfolder, fname='Lum')

        if uvspectra: calc_bias(aa, uvmesh/uvmesh.cmean(), outfolder, fname='UVbg')

        fname = 'HI_UVbg_ap%dp%d'%((profile*10)//10, (profile*10)%10)
        calc_bias(aa, h1mesh/h1mesh.cmean(), outfolder, fname=fname)

        ratio = (cencat['HIuvmass']/cencat['HImass']).compute()
        print(rank, 'Cen', '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())

        ratio = (satcat['HIuvmass']/satcat['HImass']).compute()
        print(rank, 'Sat', '%0.3f'%ratio.max(), '%0.3f'%ratio.min(), '%0.3f'%ratio.mean(), '%0.3f'%ratio.std())

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
