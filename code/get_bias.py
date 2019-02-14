import numpy as np
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog,\
                         BigFileMesh, FFTPower
from nbodykit     import setup_logging
from mpi4py       import MPI

import HImodels
# enable logging, we have some clue what's going on.
setup_logging('info')

#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc, ncsim, sim, prefix = 256, 512, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
#bs,nc,ncsim, sim, prefic = 1024, 1024, 10240, 'highres/%d-9100-fixed'%ncsim, 'highres'


# It's useful to have my rank for printing...
pm   = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank
comm = pm.comm


#Which model to use
HImodel = HImodels.ModelA
ofolder = '../data/outputs/'




def calc_bias(aa,h1mesh,suff):
    '''Compute the bias(es) for the HI'''
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
    pkh1h1 = pkh1h1['power']-pkh1h1.attrs['shotnoise']
    pkh1mm = FFTPower(h1mesh,second=dm,mode='1d').power['power']
    if rank==0: print('Done.')
    # Compute the biases.
    b1x = np.abs(pkh1mm/(pkmm+1e-10))
    b1a = np.abs(pkh1h1/(pkmm+1e-10))**0.5
    if rank==0: print("Finishing processing a={:.4f}.".format(aa))
    return(k,b1x,b1a,np.abs(pkmm))
    #


    
    

if __name__=="__main__":
    #satsuff='-m1_5p0min-alpha_0p8-16node'
    suff='-m1_00p3mh-alpha-0p8-subvol'
    outfolder = ofolder + suff[1:] + "/modelA/"
    try:  os.makedirs(outfolder)
    except : pass

    if rank==0:
        print('Starting')
    for aa in alist:


        if rank == 0: print('\n ############## Redshift = %0.2f ############## \n'%(1/aa-1))
        halocat = BigFileCatalog(scratchyf + sim+ '/fastpm_%0.4f//'%aa, dataset='LL-0.200')
        mp = halocat.attrs['MassTable'][1]*1e10##
        halocat['Mass'] = halocat['Length'].compute() * mp
        cencat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/cencat'%aa+suff)
        satcat = BigFileCatalog(scratchcm + sim+'/fastpm_%0.4f/satcat'%aa+suff)
        rsdfac = read_conversions(scratchyf + sim+'/fastpm_%0.4f/'%aa)
        #

        HImodelz = HImodel(aa)
        los = [0,0,1]
        halocat['HImass'], cencat['HImass'], satcat['HImass'] = HImodelz.assignHI(halocat, cencat, satcat)
        halocat['RSDpos'], cencat['RSDpos'], satcat['RSDpos'] = HImodelz.assignrsd(rsdfac, halocat, cencat, satcat, los=los)

        h1mesh = HImodelz.createmesh(bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass')

        kk,b1x,b1a,pkmm = calc_bias(aa,h1mesh,suff)

        #
        if rank==0:
            fout = open(outfolder + "HI_bias_{:6.4f}.txt".format(aa),"w")
            fout.write("# Mcut={:12.4e}Msun/h.\n".format(mcut))
            fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                       format("k","b1_x","b1_a","Pkmm"))
            for i in range(1,kk.size):
                fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                           format(kk[i],b1x[i],b1a[i],pkmm[i]))
            fout.close()
    #
