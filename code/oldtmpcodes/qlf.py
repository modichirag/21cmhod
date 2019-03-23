import numpy as np
from scipy.optimize import minimize, Bounds
from nbodykit.lab import BigFileCatalog, BigFileMesh



h = 0.6776

def moster(Mhalo,z,h=0.6776, scatter=None):
    Minf = Mhalo/h
    zzp1  = z/(1+z)
    M1    = 10.0**(11.590+1.195*zzp1)
    mM    = 0.0351 - 0.0247*zzp1
    beta  = 1.376  - 0.826*zzp1
    gamma = 0.608  + 0.329*zzp1
    Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
    Mstar*= Minf
    if scatter is not None: 
        Mstar = 10**(np.log10(Mstar) + np.random.normal(0, scatter, Mstar.size))
    return Mstar*h
    #                                                                                                                                                                                                                                                                                     

def qlf(M, zz, alpha=-2.03, beta=-4, Ms=-27.21, lphi6 = -8.94):
    phis = 10**(lphi6 -0.47*(zz-6))
    f1 = 10**(0.4*(alpha+1)*(M - Ms))
    f2 = 10**(0.4*(beta+1)*(M - Ms))
    return phis/(f1+f2)


def mbh(mg, alpha=-3.5, beta=1, scatter=False):
    m = mg/h
    mb = 1e10 * 10**alpha * (m/1e10)**beta
    if scatter: mb = 10**(np.log10(mb) + np.random.normal(scale=scatter, size=mb.size))
    return mb*h

def lq(mb, eta=0.1, scatter=False):
    m = mb/h
    lsun = 3.28e26
    if scatter: eta = np.random.lognormal(eta, scatter, m.size)
    return 3.3e4*eta*m *lsun

zz = 5
aa = 1/(1+zz)
dpath = '/project/projectdirs/m3127/H1mass/'
scratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
bs = 256
sim = '/highres/%d-9100-fixed'%2560


print('Reading files')
halos = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/halocat/'%aa)
cencat = BigFileCatalog(scratch + sim+ '/fastpm_%0.4f/cencat-m1_00p3mh-alpha-0p8-subvol/'%aa)
satcat = BigFileCatalog(scratch  + sim+ '/fastpm_%0.4f/satcat-m1_00p3mh-alpha-0p8-subvol/'%aa)
hpos = halos['Position'].compute()
hmass = halos['Mass'].compute()
cmass = cencat['Mass'].compute()
smass = satcat['Mass'].compute()


allmass = np.concatenate((cmass, smass))
mgal = moster(allmass, zz, scatter=0.3)
mgalshuffle = np.random.permutation(mgal)




magbins = np.linspace(-30, -10, 42)
def qlftomin(p):
    fon, alpha, beta = p
    mgalshuffle = np.random.permutation(mgal)
    mgalon = mgalshuffle[:int(fon*mgal.size)]
    mb = mbh(mgalon, scatter=0.3, alpha=alpha, beta=beta)
    lum = lq(mb, eta=0.1, scatter=0.3)
    mag14 = 72.5+0.29- 2.5*np.log10(lum)

    nmag14, xmag14 = np.histogram(mag14, magbins)
    xmag14 = -(xmag14[:-1]*xmag14[1:])**0.5
    nmag14 = nmag14/np.diff(magbins)[0]
    lmag14, _ = np.histogram(mag14, magbins, weights=lum/lum.sum())
    lmag14 = lmag14/np.diff(magbins)[0]

    ntrue = qlf(xmag14, zz)*(bs/h)**3
    
    chisq = ((ntrue - nmag14)**2 * lmag14).sum()
    #chisq = ((ntrue - nmag14)**2).sum()

    if (fon > 5e-2) or (fon < 1e-3) : chisq *= 1e10
    if (alpha > -1) or (alpha < -5) : chisq *= 1e10
    if (beta > 1.5) or (beta < 0.5) : chisq *= 1e10
    return chisq


niter = 0

def callback(xk):
    global niter
    if niter%10==0:
        print('For iteration : ', niter)
        print(xk, '%0.2e'%qlftomin(xk))
    niter +=1

print('Starting minimization')

p0 = [1e-2, -3.5, 1]
bounds = Bounds([1e-3, -5, 0.5], [5e-2, -2, 1.5])

xx = minimize(qlftomin, p0, method='Nelder-Mead', callback=callback, options={'maxfev':1000})
#xx = minimize(qlftomin, p0, callback=callback, options={'maxfev':1000}, bounds=bounds)
#xx = minimize(qlftomin, p0, method='BFGS', callback=callback)

print(xx)
