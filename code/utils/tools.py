import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower
from nbodykit.source.mesh.field import FieldMesh
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.misc import derivative

def atoz(a): return 1/a - 1
def ztoa(z): return 1/(z+1)


project = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
scratch = '/global/cscratch1/sd/yfeng1/m3127/'




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
#


def readincatalog(aa, matter=False):

    if matter: dmcat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='1')
    halocat = BigFileCatalog(scratch + sim + '/fastpm_%0.4f/'%aa, dataset='LL-0.200')
    mp = halocat.attrs['MassTable'][1]*1e10
    print('Mass of particle is = %0.2e'%mp)
    halocat['Mass'] = halocat['Length'] * mp
    halocat['Position'] = halocat['Position']%bs # Wrapping positions assuming periodic boundary conditions
    if matter:
        return dmcat, halocat
    else: return halocat



def readinhalos(aafiles, sim, dpath=project, HI_hod=HI_hod):
    aafiles = np.array(aafiles)
    zzfiles = np.array([round(atoz(aa), 2) for aa in aafiles])
    hpos, hmass, hid, h1mass = {}, {}, {}, {}
    size = {}
    
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        halos = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/halocat/'%aa)
        hmass[zz] = halos["Mass"].compute() 
        if HI_hod is not None: h1mass[zz] = HI_hod(halos['Mass'].compute(), aa)
        hpos[zz] = halos['Position'].compute()
        hid[zz] = np.arange(hmass[zz].size).astype(int)
        size[zz] = halos.csize
    return hpos, hmass, hid, h1mass, size



def readincentrals(aafiles, suff, sim, dpath=myscratch, HI_hod=HI_hod):
    aafiles = np.array(aafiles)
    zzfiles = np.array([round(atoz(aa), 2) for aa in aafiles])

    cpos, cmass, ch1mass, chid = {}, {}, {}, {}
    size = {}
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        cen = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/cencat'%aa+suff)
        cmass[zz] = cen["Mass"].compute()
        if HI_hod is not None: ch1mass[zz] = HI_hod(cen['Mass'].compute(), aa)
        cpos[zz] = cen['Position'].compute()
        chid[zz] = cen['GlobalID'].compute()
        size[zz] = cen.csize
    return cpos, cmass, chid, ch1mass, size



def readinsatellites(aafiles, suff, sim, dpath=myscratch, HI_hod=HI_hod):
    aafiles = np.array(aafiles)
    zzfiles = np.array([round(atoz(aa), 2) for aa in aafiles])

    spos, smass, sh1mass, shid = {}, {}, {}, {}
    size = {}
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        sat = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/satcat'%aa+suff)
        smass[zz] = sat["Mass"].compute()
        if HI_hod is not None: sh1mass[zz] = HI_hod(sat['Mass'].compute(), aa)
        spos[zz] = sat['Position'].compute()
        shid[zz] = sat['GlobalID'].compute()
        size[zz] = sat.csize
    return spos, smass, shid, sh1mass, size




def loginterp(x, y, yint = None, side = "both", lorder = 15, rorder = 15, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6, k=5):

    if yint is None:
        yint = ius(x, y, k = k)

    if side == "both":
        side = "lr"
        l =lp
        r =rp
    lneff, rneff = 0, 0
    niter = 0
    while (lneff <= 0) or (lneff > 1):
        lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
        l +=1
        if niter > 100: continue
    print('Left slope = %0.3f at point '%lneff, l)
    niter = 0
    while (rneff < -3) or (rneff > -2):
        rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
        r -= 1
        if niter > 100: continue
    print('Rigth slope = %0.3f at point '%rneff, r)

    xl = np.logspace(-18, np.log10(x[l]), 10**6.)
    xr = np.logspace(np.log10(x[r]), 10., 10**6.)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = np.concatenate((xl, xint))
        yint = np.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = np.concatenate((xint, xr))
        yint = np.concatenate((yint, yr))
    yint2 = ius(xint, yint, k = k)

    return yint2
