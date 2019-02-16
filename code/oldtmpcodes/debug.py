import numpy as np
from time import time
import re
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog, FFTPower
#
#
#Global, fixed things
scratch1 = '/global/cscratch1/sd/yfeng1/m3127/'
scratch2 = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
#alist    = [0.2222,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc = 256, 512
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'
bs,nc,ncsim = 1024, 1024, 10240
sim,prefix  = 'highres/%d-9100-fixed'%ncsim, 'highres'
pm = ParticleMesh(BoxSize=bs, Nmesh=[nc, nc, nc])
rank = pm.comm.rank


# This should be imported once from a "central" place.
def HI_hod(mhalo,aa):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = 1.0
    alp = (1+2*zz)/(2+2*zz)
    mcut= 1e9*( 1.8 + 15*(3*aa)**8 )
    norm= 3e5*(1+(3.5/zz)**6)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #



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



def calc_pkhalo(aa, suff):
    if rank ==0 : print('Read in central/satellite catalogs')
    #cencat = BigFileCatalog(project+sim+'/fastpm_%0.4f/cencat'%aa)
    cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat-16node'%aa)
    cencat = BigFileCatalog(scratch1+sim+'/fastpm_%0.4f/'%aa+\
                        'LL-0.200','')
    rsdfac = read_conversions(scratch1+sim+'/fastpm_%0.4f/'%aa)
    # Compute the power spectrum
    los = [0,0,1]
    cpos = cencat['Position'].compute()
    start = time()

    cenmesh = pm.paint(cencat['Position'])
    h1mesh = cenmesh
    end = time()
    if rank ==0 : print('Time taken to mesh = ', end - start)
    start = time()
    #pkh1h1   = FFTPower(h1mesh,mode='1d',kmin=0.025,dk=0.0125).power
    pkh1h1   = FFTPower(h1mesh/h1mesh.cmean(),mode='1d').power
    end = time()
    if rank ==0 : print('Time taken to power = ', end - start)
    # Extract the quantities we want and write the file.
    kk   = pkh1h1['k']
    sn   = pkh1h1.attrs['shotnoise']
    pk   = np.abs(pkh1h1['power'])
    if rank ==0 : 
        fout = open("../data/L%04d/Halocat_pks_1d_%0.4f.txt"%(bs, aa),"w")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_0_HI"))
        for i in range(kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]-sn))
        fout.close()
        #









def calc_bias(aa,mcut,suff):
    '''Compute the bias(es) for the HI'''
    if rank==0:
        print("Processing a={:.4f}...".format(aa))
        print('Reading DM mesh...')
    dm    = BigFileMesh(scratch1+sim+'/fastpm_%0.4f/'%aa+\
                        '/1-mesh/N%04d'%nc,'').paint()
    dm   /= dm.cmean()
    if rank==0: print('Computing DM P(k)...')
    pkmm  = FFTPower(dm,mode='1d').power
    k,pkmm= pkmm['k'],pkmm['power']  # Ignore shotnoise.
    if rank==0: print('Done.')
    #
    if rank==0: print('Reading central/satellite catalogs...')
    cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat-16node'%aa)
#    satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+suff)
#    if rank==0: print('Catalogs read.')
#    #
#    if rank==0: print('Computing HI masses...')
#    cencat['HImass'] = HI_hod(cencat['Mass'],aa,mcut)   
#    satcat['HImass'] = HI_hod(satcat['Mass'],aa,mcut)   
#    totHImass        = cencat['HImass'].sum().compute() +\
#                       satcat['HImass'].sum().compute()
#    cencat['HImass']/= totHImass/float(nc)**3
#    satcat['HImass']/= totHImass/float(nc)**3
#    if rank==0: print('HI masses done.')
#    #
#    if rank==0: print('Combining catalogs and computing P(k)...')
#    allcat = MultipleSpeciesCatalog(['cen','sat'],cencat,satcat)
#    #h1mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],weight='HImass')

    cenmesh = pm.paint(cencat['Position'])
    #satmesh = pm.paint(satcat['Position'], mass=satcat['HImass'])
    h1mesh = cenmesh 
    pkh1h1 = FFTPower(h1mesh,mode='1d').power
    pkh1h1 = pkh1h1['power']-pkh1h1.attrs['shotnoise']
    pkh1mm = FFTPower(h1mesh,second=dm,mode='1d').power['power']
    if rank==0: print('Done.')
    # Compute the biases.
    b1x = np.abs(pkh1mm/(pkmm+1e-10))
    b1a = np.abs(pkh1h1/(pkmm+1e-10))**0.5
    if rank==0: print("Finishing processing a={:.4f}.".format(aa))
    if rank==0:
        fout = open("../data/L%04d/HI_bias_halo_%0.4f.txt"%(bs, aa),"w")
        fout.write("# Mcut={:12.4e}Msun/h.\n".format(mcut))
        fout.write("# {:>8s} {:>10s} {:>10s} {:>15s}\n".\
                   format("k","b1_x","b1_a","Pkmm"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f} {:15.5e}\n".\
                       format(kk[i],b1x[i],b1a[i],pkmm[i]))
        fout.close()
        bavg = np.mean(b1x[1:10])
        flog.write("{:6.2f} {:12.4e} {:6.3f}\n".format(1/aa-1,mcut,bavg))
        flog.flush()
    if rank==0: flog.close()
    #
    return(k,b1x,b1a,np.abs(pkmm))
    #


    
    

def calc_pk1d(aa,suff):
    '''Compute the 1D redshift-space P(k) for the HI'''
    if rank ==0 : print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat-16node'%aa)
    satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    rsdfac = read_conversions(scratch1+sim+'/fastpm_%0.4f/'%aa)
    # Compute the power spectrum
    los = [0,0,1]
    cpos = cencat['Position'].compute()
    print(rank, cpos.max(), cpos.min())
    print(rank, cpos)

    cencat['RSDpos'] = cencat['Position']+cencat['Velocity']*los * rsdfac
    satcat['RSDpos'] = satcat['Position']+satcat['Velocity']*los * rsdfac
    start = time()
    cencat['HImass'] = HI_hod(cencat['Mass'],aa)
    satcat['HImass'] = HI_hod(satcat['Mass'],aa)
    totHImass        = cencat['HImass'].sum().compute() +\
                       satcat['HImass'].sum().compute()
    end = time()
    if rank ==0 : print('Time taken to assign mass = ', end - start)
    
    cencat['HImass']/= totHImass/float(nc)**3
    satcat['HImass']/= totHImass/float(nc)**3
    allcat  = MultipleSpeciesCatalog(['cen','sat'],cencat,satcat)
    start = time()
    h1mesh  = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                             position='RSDpos',weight='HImass')
    #pkh1h1   = FFTPower(h1mesh,mode='1d',kmin=0.025,dk=0.0125).power

    cenmesh = pm.paint(cencat['RSDpos'], mass=cencat['HImass'])
    satmesh = pm.paint(satcat['RSDpos'], mass=satcat['HImass'])
    h1mesh = cenmesh + satmesh
    end = time()
    if rank ==0 : print('Time taken to mesh = ', end - start)
    start = time()
    #pkh1h1   = FFTPower(h1mesh,mode='1d',kmin=0.025,dk=0.0125).power
    pkh1h1   = FFTPower(h1mesh,mode='1d').power
    end = time()
    if rank ==0 : print('Time taken to power = ', end - start)
    # Extract the quantities we want and write the file.
    kk   = pkh1h1['k']
    sn   = pkh1h1.attrs['shotnoise']
    pk   = np.abs(pkh1h1['power'])
    if rank ==0 : 
        fout = open("../data/HI_pks_1d_{:6.4f}.txt".format(aa),"w")
        fout.write("# Subtracting SN={:15.5e}.\n".format(sn))
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_0_HI"))
        for i in range(kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]-sn))
        fout.close()
        #




def calc_pkmu(aa,suff):
    '''Compute the redshift-space P(k) for the HI in mu bins'''
    if rank ==0 : print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat-16node'%aa)
    satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    rsdfac = read_conversions(scratch1+sim+'/fastpm_%0.4f/'%aa)
    # Compute P(k,mu).
    los = [0,0,1]
    cencat['RSDpos'] = cencat['Position']+cencat['Velocity']*los * rsdfac
    satcat['RSDpos'] = satcat['Position']+satcat['Velocity']*los * rsdfac
    cencat['HImass'] = HI_hod(cencat['Mass'],aa)
    satcat['HImass'] = HI_hod(satcat['Mass'],aa)
    totHImass        = cencat['HImass'].sum().compute() +\
                       satcat['HImass'].sum().compute()
    cencat['HImass']/= totHImass/float(nc)**3
    satcat['HImass']/= totHImass/float(nc)**3
    allcat = MultipleSpeciesCatalog(['cen','sat'],cencat,satcat)
    h1mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                             position='RSDpos',weight='HImass')
    #pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=4,los=los).power

    cenmesh = pm.paint(cencat['RSDpos'], mass=cencat['HImass'])
    satmesh = pm.paint(satcat['RSDpos'], mass=satcat['HImass'])
    h1mesh = cenmesh + satmesh

    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=4,los=los).power
    # Extract what we want.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    pk = pkh1h1['power']
    # Write the results to a file.
    if rank ==0 : 
        fout = open("../data/HI_pks_mu_{:06.4f}.txt".format(aa),"w")
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


    



def calc_pkll(aa,suff):
    '''Compute the redshift-space P_ell(k) for the HI'''
    if rank ==0 : print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/cencat-16node'%aa)
    satcat = BigFileCatalog(scratch2+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    rsdfac = read_conversions(scratch1+sim+'/fastpm_%0.4f/'%aa)
    #
    los = [0,0,1]
    cencat['RSDpos'] = cencat['Position']+cencat['Velocity']*los * rsdfac
    satcat['RSDpos'] = satcat['Position']+satcat['Velocity']*los * rsdfac
    cencat['HImass'] = HI_hod(cencat['Mass'],aa)
    satcat['HImass'] = HI_hod(satcat['Mass'],aa)
    totHImass        = cencat['HImass'].sum().compute() +\
                       satcat['HImass'].sum().compute()
    cencat['HImass']/= totHImass/float(nc)**3
    satcat['HImass']/= totHImass/float(nc)**3
    allcat = MultipleSpeciesCatalog(['cen','sat'],cencat,satcat)

    h1mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                             position='RSDpos',weight='HImass')
    #pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=8,los=los,poles=[0,2,4]).poles
    
    cenmesh = pm.paint(cencat['RSDpos'], mass=cencat['HImass'])
    satmesh = pm.paint(satcat['RSDpos'], mass=satcat['HImass'])
    h1mesh = cenmesh + satmesh

    pkh1h1 = FFTPower(h1mesh,mode='2d',Nmu=8,los=los,poles=[0,2,4]).poles
    # Extract the quantities of interest.
    kk = pkh1h1.coords['k']
    sn = pkh1h1.attrs['shotnoise']
    P0 = pkh1h1['power_0'].real - sn
    P2 = pkh1h1['power_2'].real
    P4 = pkh1h1['power_4'].real
    # Write the results to a file.
    if rank ==0 : 
        fout = open("../data/HI_pks_ll_{:06.4f}.txt".format(aa),"w")
        fout.write("# Redshift space power spectrum multipoles.\n")
        fout.write("# Subtracting SN={:15.5e} from monopole.\n".format(sn))
        fout.write("# {:>8s} {:>15s} {:>15s} {:>15s}\n".format("k","P0","P2","P4"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:15.5e} {:15.5e} {:15.5e}\n".\
                       format(kk[i],P0[i],P2[i],P4[i]))
        fout.close()
    #


    
    

if __name__=="__main__":
    print('Starting')
    satsuff='-mmin0p1_m1_5p0min-alpha_0p9'
    satsuff='-m1_8p0min-alpha_0p9'
    satsuff='-m1_5p0min-alpha_0p9'
    satsuff='-m1_5p0min-alpha_0p8-16node'
    for aa in alist:
        calc_pkhalo(aa,satsuff)
        #calc_bias(aa,mcut=1, suff=satsuff)
#        calc_pk1d(aa,satsuff)
#        calc_pkmu(aa,satsuff)
#        calc_pkll(aa,satsuff)
#    #
