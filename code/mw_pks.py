import numpy as np
import re
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, FFTPower
#
#
#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist   = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
alist   = [0.2500]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc = 256, 512
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'




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





def calc_pk(aa,suff):
    '''Compute the redshift-space P(k) for the HI'''
    print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(project+sim+'/fastpm_%0.4f/cencat'%aa)
    satcat = BigFileCatalog(project+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    rsdfac = read_conversions(scratch+sim+'/fastpm_%0.4f/'%aa)
    print("RSD factor=",rsdfac)
    #
    los      = [0,0,rsdfac]
    cpos     = cencat['Position']+cencat['VelocityOffset']*los
    cmass    = cencat['Mass']
    spos     = satcat['Position']+satcat['VelocityOffset']*los
    smass    = satcat['Mass']
    pos      = np.concatenate((cpos,spos),axis=0)
    ch1mass  = HI_hod(cmass,aa)   
    sh1mass  = HI_hod(smass,aa)   
    h1mass   = np.concatenate((ch1mass,sh1mass),axis=0)
    pm       = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
    h1mesh   = pm.paint(pos,mass=h1mass)    
    pkh1h1   = FFTPower(h1mesh/h1mesh.cmean(),mode='1d').power
    # Extract the quantities we want.
    kk  = pkh1h1['k']
    pk  = pkh1h1['power']
    return(kk,np.abs(pk))
    #




def calc_pkmu(aa,suff):
    '''Compute the redshift-space P(k) for the HI in mu bins'''
    print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(project+sim+'/fastpm_%0.4f/cencat'%aa)
    satcat = BigFileCatalog(project+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    rsdfac = read_conversions(scratch+sim+'/fastpm_%0.4f/'%aa)
    print("RSD factor=",rsdfac)
    #
    los      = [0,0,rsdfac]
    cpos     = cencat['Position']+cencat['Velocity']*los
    cmass    = cencat['Mass']
    spos     = satcat['Position']+satcat['Velocity']*los
    smass    = satcat['Mass']
    pos      = np.concatenate((cpos,spos),axis=0)
    ch1mass  = HI_hod(cmass,aa)   
    sh1mass  = HI_hod(smass,aa)   
    h1mass   = np.concatenate((ch1mass,sh1mass),axis=0)
    pm       = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
    h1mesh   = pm.paint(pos,mass=h1mass)    
    pkh1h1   = FFTPower(h1mesh/h1mesh.cmean(),mode='2d',Nmu=5,los=los).power
    # Write the results to a file.
    fout = open("HI_pks_mu_{:06.4f}.txt".format(aa),"w")
    fout.write("# Redshift space power spectrum in mu bins.\n")
    ss = "# {:>6s}".format(r'k\mu')
    for i in range(pkh1h1.shape[1]):
        ss += " {:15.4f}".format(pkh1h1.coords['mu'][i])
    fout.write(str+"\n")
    for i in range(pkh1h1.shape[0]):
        ss = "{:8.4f}".format(pkh1h1['k'][i])
        for j in range(pkh1h1.shape[1]):
            ss += " {:15.5e}".format(pkh1h1[i,j]['power'].real)
        fout.write(str+"\n")
    fout.close()
    #


    
    

if __name__=="__main__":
    print('Starting')
    satsuff='-mmin0p1_m1_5p0min-alpha_0p9'
    satsuff='-m1_8p0min-alpha_0p9'
    satsuff='-m1_5p0min-alpha_0p9'
    for aa in alist:
        zz   = 1.0/aa-1.0
        kk,pk= calc_pk(aa,satsuff)
        #
        fout = open("HI_pks_1d_{:6.4f}.txt".format(aa),"w")
        fout.write("# {:>8s} {:>15s}\n".format("k","Pk_HI"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:15.5e}\n".format(kk[i],pk[i]))
        fout.close()
        calc_pkmu()
    #
