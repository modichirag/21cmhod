import numpy as np
from pmesh.pm     import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
#
#
#Global, fixed things
scratch = '/global/cscratch1/sd/yfeng1/m3127/'
project = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
aafiles = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
alist   = [0.1429,0.1667,0.2000,0.2500,0.2857,0.3333]


#Parameters, box size, number of mesh cells, simulation, ...
bs, nc = 256, 512
ncsim, sim, prefix = 2560, 'highres/%d-9100-fixed'%2560, 'highres'




def HI_hod(mhalo,aa,mcut=2e9):
    """Returns the 21cm "mass" for a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    alp = (1+2*zz)/(2+2*zz)
    norm= 2e9*np.exp(-1.9*zp1+0.07*zp1**2)
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    mHI*= norm
    return(mHI)
    #





def calc_bias(aa,mcut,suff):
    '''Compute the bias(es) for the HI'''
    print('Read in DM mesh')
    dm = BigFileMesh(project+sim+'/fastpm_%0.4f/'%aa+\
                     '/dmesh_N%04d'%nc,'1').paint()
    pkmm  = FFTPower(dm/dm.cmean(), mode='1d').power
    k,pkmm= pkmm['k'],pkmm['power']
    #
    print('Read in central/satellite catalogs')
    cencat = BigFileCatalog(project+sim+'/fastpm_%0.4f/cencat'%aa)
    satcat = BigFileCatalog(project+sim+'/fastpm_%0.4f/satcat'%aa+suff)
    #
    cpos,cmass = cencat['Position'],cencat['Mass']
    spos,smass = satcat['Position'],satcat['Mass']
    pos        = np.concatenate((cpos,spos),axis=0)
    ch1mass    = HI_hod(cmass,aa,mcut)   
    sh1mass    = HI_hod(smass,aa,mcut)   
    h1mass     = np.concatenate((ch1mass,sh1mass),axis=0)    
    pm         = ParticleMesh(BoxSize=bs,Nmesh=[nc,nc,nc])
    h1mesh     = pm.paint(pos,mass=h1mass)    
    pkh1h1     = FFTPower(h1mesh/h1mesh.cmean(),mode='1d').power['power']
    pkh1mm     = FFTPower(h1mesh/h1mesh.cmean(),\
                          second=dm/dm.cmean(),mode='1d').power['power']
    # Compute the biases.
    b1x = np.abs(pkh1mm/(pkmm+1e-10))
    b1a = np.abs(pkh1h1/(pkmm+1e-10))**0.5
    return(k,b1x,b1a)
    #


    
    

if __name__=="__main__":
    print('Starting')
    satsuff='-m1_5p0min-alpha_0p9' # '-m1_8p0min-alpha_0p9'
    flog = open("HI_bias_vs_z.txt","w")
    flog.write("# {:>4s} {:>12s} {:>6s}\n".format("z","Mcut","b"))
    for aa in alist:
        zz         = 1.0/aa-1.0
        mcut       = 2e10*aa
        mcut       = 1e10*np.exp(-(zz-2.0))
        mcut       = 1.5e10*(3*aa)**3
        mcut       = 5e9
        mcut       = 1e9*( 1.8 + 15*(3*aa)**8 )
        kk,b1x,b1a = calc_bias(aa,mcut,satsuff)
        #
        fout = open("HI_bias_{:6.4f}.txt".format(aa),"w")
        fout.write("# Mcut={:12.4e}Msun/h.\n".format(mcut))
        fout.write("# {:>8s} {:>10s} {:>10s}\n".format("k","b1_x","b1_a"))
        for i in range(1,kk.size):
            fout.write("{:10.5f} {:10.5f} {:10.5f}\n".\
                       format(kk[i],b1x[i],b1a[i]))
        fout.close()
        #
        bavg = np.mean(b1x[1:6])
        flog.write("{:6.2f} {:12.4e} {:6.3f}\n".format(1/aa-1,mcut,bavg))
        flog.flush()
    flog.close()
    #
