import numpy as np
import matplotlib.pyplot as plt



# Things to change
betaf = lambda zz: (1+2*zz)/(2+2*zz)
mcutf = lambda zz : 1e9*( 1.8 + 15*(3/(zz+1))**8 )
normf = lambda zz : 3e5*(1+(3.5/zz)**6)

betaf2 = lambda zz: (1+3*zz)/(4+3*zz)
#betaf2 = lambda zz: (1+2*zz)/(2+2*zz)/1.2
mcutf2 = lambda zz : 1e9*( 1.8 + 15*(3/(zz+1))**8 )
normf2 = lambda zz : 3e5*(1+(3.5/zz)**6)


f = 0.05
alpha = -0.9

#######################################################
mh = np.logspace(9, 14)
aafiles = np.array([0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333])
zzfiles = np.array([round(1/aa-1, 2) for aa in aafiles])



def HIm(mhalo, mcut, beta, A):
    """Returns the 21cm "mass" for a box of halo masses."""
    xx  = mhalo/mcut+1e-10
    mHI = xx**beta * np.exp(-1/xx)
    mHI*= A
    return(mHI)


def nsat(f, mh, alpha, mmin=None, mcut=None):
    if mmin == None and mcut==None:
        print('Give mmnin of satellites')
        return None
    elif mcut is not None: mmin = 0.1*mcut + 0*mh
    return (mmin/f/mh)**alpha

def mtotsat2(f, mh, alpha, mmax=None, mmin=None, mcut=None):
    if mmin == None and mcut==None:
        print('Give mmnin of satellites')
        return None
    elif mcut is not None: mmin = 0.1*mcut + 0*mh
    if mmax is None: mmax = mh/10.

    mask = mmin > mmax/2
    mmin[mask] = mmax[mask]/2
    return -alpha/(alpha+1)/f**alpha *((mmax/mh)**(alpha + 1) - (mmin/mh)**(alpha + 1))*mh


def mtotsat(f, mh, alpha, mmax=None, mmin=None, mcut=None):
    if mmin == None and mcut==None:
        print('Give mmnin of satellites')
        return None
    elif mcut is not None: mmin = 0.1*mcut + 0*mh
    if mmax is None: mmax = mh/10.

    toret = np.zeros_like(mh)
    fac = -alpha/f**alpha/mh**alpha
    for i in range(toret.size):
        if mmin[i]  > mmax[i]/2 : mmini = mmax[i]/2
        else: mmini = mmin[i]
        mm = np.logspace(np.log10(mmini), np.log10(mmax[i]), 1000)
        y = mm**(alpha )
        toret[i] = fac[i] * np.trapz(y, mm)
    return toret

def mtotsatHI(f, mh, mcut, beta, A, alpha, mmax=None, mmin=None):
    if mmin == None and mcut==None:
        print('Give mmnin of satellites')
        return None
    elif mcut is not None: mmin = 0.1*mcut + 0*mh
    if mmax is None: mmax = mh/10.

    toret = np.zeros_like(mh)
    fac = -alpha/f**alpha/mh**alpha *A/mcut**beta
    for i in range(toret.size):
        if mmin[i]  > mmax[i]/2 : mmini = mmax[i]/2
        else: mmini = mmin[i]
        mm = np.logspace(np.log10(mmini), np.log10(mmax[i]), 10000)
        y = mm**(alpha + beta - 1)* np.exp(-mcut/mm)
        toret[i] = fac[i] * np.trapz(y, mm)
    return toret




fig, ax = plt.subplots(1, 4, figsize = (15, 3))
ax[0].plot(zzfiles, betaf(zzfiles))
ax[0].plot(zzfiles, betaf2(zzfiles), '--')
ax[0].set_title('beta')
ax[1].plot(zzfiles, mcutf(zzfiles))
ax[1].plot(zzfiles, mcutf2(zzfiles), '--')
ax[1].set_yscale('log')
ax[1].set_title('mcut')
ax[2].plot(zzfiles, normf(zzfiles))
ax[2].plot(zzfiles, normf2(zzfiles), '--')
ax[2].set_yscale('log')
ax[2].set_title('norm')
for iz, zz in enumerate(zzfiles): 
    ax[3].plot(mh, 1+HIm(mh, mcutf(zz), betaf(zz), normf(zz) ), 'C%d'%iz)
    ax[3].plot(mh, 1+HIm(mh, mcutf2(zz), betaf2(zz), normf2(zz) ), 'C%d--'%iz)
ax[3].loglog('log')
ax[3].set_ylim(1e5, 1e10)
ax[3].set_title('MHI')
for axis in ax: axis.grid(which='both')
plt.savefig('./figs/playh1hod.png')
plt.show()



fig, ax = plt.subplots(3, 3, figsize = (12, 12))

for iz, zz in enumerate(zzfiles):
    aa = 1/(zz+1)
    beta = betaf2(zz)
    mcut = mcutf2(zz)
    norm = normf2(zz)

    axis = ax.flatten()[iz]

    mtot = mtotsat(f, mh, alpha, mcut=mcut)
    mHIsat = mtotsatHI(f, mh, mcut=mcut, beta=beta, A=norm, alpha=alpha) 
    mHIcen = HIm(mh-mtot, mcut=mcut, beta=beta, A=norm)
    mHIhalo = HIm(mh, mcut=mcut, beta=beta, A=norm)

    axis.plot(mh, mtot/mh, 'k:', lw=2, label='sat mass frac')
    axis.plot(mh, mHIsat/(mHIcen+mHIsat), 'r', lw=1.5, label='sat HI frac of galaxy')
    axis.plot(mh, mHIcen/(mHIcen+mHIsat), 'b', lw=1.5, label='cen HI frac of galaxy')
    axis.plot(mh, mHIsat/mHIhalo, 'r--', lw=3, alpha=0.7, label='sat HI frac of Halo')
    axis.plot(mh, mHIcen/mHIhalo, 'b--', lw=3, alpha=0.7, label='sat HI frac of Halo')
    axis.set_xscale('log')
    axis.set_title('z=%.1f'%zz)

    axis.set_xscale('log')
    axis.set_title('z=%.1f'%zz)

for axis in ax.flatten(): axis.grid(which='both')
lgd = ax[0, 2].legend(loc='lower right', bbox_to_anchor=(1, 1.1), ncol=3, fontsize=13)
fig.savefig('./figs/playh1frac.png', bbox_extra_artists=(lgd,))

##
fig, ax = plt.subplots(3, 3, figsize = (12, 12))
for iz, zz in enumerate(zzfiles):
    aa = 1/(zz+1)
    beta = betaf(zz)
    mcut = mcutf(zz)
    norm = normf(zz)

    axis = ax.flatten()[iz]

    mtot = mtotsat(f, mh, alpha, mcut=mcut)
    mHIsat = mtotsatHI(f, mh, mcut=mcut, beta=beta, A=norm, alpha=alpha) 
    mHIcen = HIm(mh-mtot, mcut=mcut, beta=beta, A=norm)
    mHIhalo = HIm(mh, mcut=mcut, beta=beta, A=norm)

    axis.plot(mh, mtot/mh, 'k:', lw=1.5, label='sat mass frac')
    axis.plot(mh, mHIsat/(mHIcen+mHIsat), 'm', lw=1.2, label='sat HI frac of galaxy')
    axis.plot(mh, mHIcen/(mHIcen+mHIsat), 'g', lw=1.2, label='cen HI frac of galaxy')
    axis.plot(mh, mHIsat/mHIhalo, 'm--', lw=1.5, alpha=0.7, label='sat HI frac of halo')
    axis.plot(mh, mHIcen/mHIhalo, 'g--', lw=1.5, alpha=0.7, label='cen HI frac of halo')
    axis.set_xscale('log')
    axis.set_title('z=%.1f'%zz)

for axis in ax.flatten(): axis.grid(which='both')
lgd = ax[0, 2].legend(loc='lower right', bbox_to_anchor=(1, 1.1), ncol=3, fontsize=13)
fig.savefig('./figs/fidh1frac.png', bbox_extra_artists=(lgd,))

##
fig2, ax2 = plt.subplots()
for iz, zz in enumerate(zzfiles):
    beta = betaf2(zz)
    mcut = mcutf2(zz)
    norm = normf2(zz)
    ax2.plot(mh, nsat(f, mh, alpha, mcut=mcut), 'C%d-'%iz)

    beta = betaf(zz)
    mcut = mcutf(zz)
    norm = normf(zz)
    ax2.plot(mh, nsat(f, mh, alpha, mcut=mcut), 'C%d--'%iz)
ax2.loglog()
ax2.set_xlabel('Halo mass')
ax2.set_ylabel('Number of satellites')
fig2.savefig('./figs/playnsat.png')

print('Played')
