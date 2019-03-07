import numpy as np
from scipy.integrate import simps

def mtotsatHImodelA(mh, hodparams, h1hsatparams, mmax=None, mmin=None, mcutp=1.0):
    f, alpha = hodparams
    mcut, alphah1, A = h1hsatparams
##    if type(mh) is not np.ndarray: mh = np.array(mh).reshape(-1)
##    if mmin is None :
##        if mcut is None :
##            print('Give mmnin of satellites')
##            return None
##        elif mcut is not None: mmin = 0.1*mcut + 0*mh
##    elif type(mmin) is not np.ndarray:  mmin = mmin + 0*mh
##    if mmax is None: mmax = mh/10.

    mmax = mh/10.
    mmin = 0.1*mcut + mh*0
    
    toret = np.zeros_like(mh)
    fac = -alpha/f**alpha/mh**alpha *A/mcut**alphah1
    for i in range(toret.size):
        
        if mmin[i] > mmax[i]: continue
        else: 
            mm = np.logspace(np.log10(mmin[i]), np.log10(mmax[i]), 100)
            y = mm**(alpha + alphah1 - 1)* np.exp(-(mcut/mm)**mcutp)
            toret[i] = fac[i] * np.trapz(y, mm)
    return toret


##def mtotsatHImodelA(mh, hodparams, h1hsatparams, mmax=None, mmin=None, mcutp=1.0):
##
##    f, alpha = hodparams
##    mcut, alphah1, A = h1hsatparams
##    mh = mh.reshape(-1, 1)
##    mmin = 0.2*mcut
##    mmax = mh/10.
##    
##    fac = -alpha/f**alpha/mh**alpha *A/mcut**alphah1
##    mmaxint = mh.max()/10.
##    mm = np.logspace(np.log10(mmin), np.log10(mmaxint), 200).reshape(1, -1)
##    mask = mm <= mmax
##    y = (mm**(alpha + alphah1 - 1)* np.exp(-(mcut/mm)**mcutp))*mask
##    toret = fac * np.trapz(y, np.squeeze(mm), axis=1) 
##    return toret.flatten()
##





def nmsat(m, mh, f, alpha):
    return -alpha * m**(alpha-1) / (f*mh)**alpha


def nsattot(mh, f, alpha, mmin, mmax=None):
    if type(mh) is not np.ndarray: mh = np.array(mh).reshape(-1)
    if mmax is None: mmax = mh/10.
    if type(mmin) is not np.ndarray:  mmin = mmin + 0*mh
    mmin[mmin > mmax] = mmax[mmin > mmax]
    
    toret = np.zeros_like(mh)
    for i in range(toret.size):
        mm = np.logspace(np.log10(mmin[i]), np.log10(mmax[i]), 1000)
        y = nmsat(mm, mh[i], f, alpha)
        toret[i] = simps(y, mm)
    toret[np.isnan(toret)] = 0
    return toret


def HIm(mhalo, mcut, alphah1, A, mcutp=1.0):
    """Returns the 21cm "mass" for a box of halo masses."""
    xx  = mhalo/mcut+1e-10
    mHI = xx**alphah1 * np.exp(-1/xx**mcutp)
    mHI*= A
    return(mHI)


def massweightedsum(mh, mmin, numf, massf):
    '''
    numf gives number of objects of mass 'm' in halo of mass 'mh'
    massf gives mass weight of object of object of mass 'm'
    integration is done from mmin to mh/10
    '''
    if type(mh) is not np.ndarray: mh = np.array(mh).reshape(-1)
    if type(mmin) is not np.ndarray:  mmin = mmin + 0*mh
    mmax = mh/10.
    mmin[mmin > mmax] = mmax[mmin > mmax]
    
    toret = np.zeros_like(mh)
    for i in range(toret.size):
        mm = np.logspace(np.log10(mmin[i]), np.log10(mmax[i]), 1000)
        y = numf(mm, mh[i]) * massf(mm)
        toret[i] = np.trapz(y, mm)
    return toret
