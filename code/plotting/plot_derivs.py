#!/usr/bin/env python3
#
# Plots the power spectra for the matter and HI,
# compared to the Zeldovich/ZEFT models.
#
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append('../')
from HImodels import ModelC
from nbodykit.cosmology.cosmology import Cosmology


from matplotlib import rc, rcParams, font_manager
rcParams['font.family'] = 'serif'
fsize = 12
fontmanage = font_manager.FontProperties(family='serif', style='normal',
    size=fsize, weight='normal', stretch='normal')
font = {'family': fontmanage.get_family()[0],
        'style':  fontmanage.get_style(),
        'weight': fontmanage.get_weight(),
        'size': fontmanage.get_size(),
        }


sig80 = 0.8222
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
cosmo = Cosmology.from_dict(cosmodef)
dgrow = cosmo.scale_independent_growth_factor


dpath = '../../data/outputs/m1_00p3mh-alpha-0p8-subvol-big/ModelC/'
dpathup = '../../data/outputs/m1_00p3mh-alpha-0p8-subvol-big-up/ModelC/'
dpathdn = '../../data/outputs/m1_00p3mh-alpha-0p8-subvol-big-dn/ModelC/'
suff = 'm1_00p3mh-alpha-0p8-subvol-big'

figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass


fig, ax = plt.subplots(1, 3, figsize = (10,3))

zlist = [2, 4, 6]
for iz, zz in enumerate(zlist):
    aa = 1/(zz+1)
    mod = ModelC(zz)
    alp0, mcut0 = mod.alp, mod.mcut
    lmcut0 = np.log10(mcut0)
    mod.derivate('alpha', 0.05)
    alpd = mod.alp
    mod.derivate('mcut', 0.05)
    mcutd = mod.mcut
    mod = ModelC(zz)
    mod.derivate('mcut', -0.05)
    mcutdd = mod.mcut
    mod.derivate('alpha', -0.05)
    alpdd = mod.alp
    s8 = sig80*dgrow(zz)**1

    fidp = [s8, alp0, mcut0]
    upp = [1.05**0.5*s8, alpd, mcutd]
    dnp = [0.95**0.5*s8, alpdd, mcutdd]
    keys = ['sigma', 'alpha', 'mcut']
    fidp = {keys[i]:fidp[i] for i in range(3)}
    diffs = {keys[i]:upp[i] - dnp[i] for i in range(3)}
    diffsdn = {keys[i]:dnp[i] - fidp[keys[i]] for i in range(3)}
    diffsup = {keys[i]:upp[i] - fidp[keys[i]] for i in range(3)}

    p1 = np.loadtxt(dpath + 'mcut_vp05/HI_bias_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpath + 'mcut_vm05/HI_bias_{:06.4f}.txt'.format(aa)).T
    fid = np.loadtxt(dpath + 'HI_bias_{:06.4f}.txt'.format(aa)).T
    bb1a, bb2a = p1[2][1:6].mean(), p2[2][1:6].mean()
    bba = fid[2][1:6].mean()
    dbba = (bb1a - bb2a)/(diffs['mcut'])*fidp['mcut']/bba
    ##
    
    #Real space
    p1 = np.loadtxt(dpathup + 'matchbias_mcut/HI_bias_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpathdn + 'matchbias_mcut/HI_bias_{:06.4f}.txt'.format(aa)).T
    fid = np.loadtxt(dpath + 'HI_bias_{:06.4f}.txt'.format(aa)).T
    kk = fid[0]
    pk1 , pk2, pkf = p1[2]**2*p1[3], p2[2]**2*p2[3], fid[2]**2*fid[3]
#     pk1 *= fid[2][1]/p1[2][1]
#     pk2 *= fid[2][1]/p2[2][1]
    deriv = (pk1-pk2)/diffs['sigma']*fidp['sigma']/pkf
    ax[0].plot(kk, deriv, 'C%d-'%iz, label='z = %.1f'%zz, lw=2)

    p1 = np.loadtxt(dpath + 'mcut_vp05/HI_bias_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpath + 'mcut_vm05/HI_bias_{:06.4f}.txt'.format(aa)).T
    pk1 , pk2 = p1[2]**2*p1[3], p2[2]**2*p2[3]
    deriv = (pk1-pk2)/diffs['mcut']*fidp['mcut']/pkf
    deriv /= dbba
    ax[0].plot(kk, deriv, 'C%d--'%iz, lw=2)
    ax[0].text(0.1,3.5,"Real", fontdict=font)

    
    #l=0
    p1 = np.loadtxt(dpathup + 'matchbias_mcut/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpathdn + 'matchbias_mcut/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    fid = np.loadtxt(dpath + 'HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    kk = fid[0]
    pk1 , pk2, pkf = p1[1], p2[1], fid[1]
    deriv = (pk1-pk2)/diffs['sigma']*fidp['sigma']/pkf
    lbl = ''
    #if iz == 0: lbl = r'dlog($P$)/dlog($\sigma_8$)'
    if iz == 0: lbl = r'$\theta=\sigma_8$'
    ax[1].plot(kk, deriv, 'C%d-'%iz, label=lbl, lw=2)

    p1 = np.loadtxt(dpath + 'mcut_vp05/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpath + 'mcut_vm05/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    pk1 , pk2 = p1[1], p2[1]
    deriv = (pk1-pk2)/diffs['mcut']*fidp['mcut']/pkf
    deriv /= dbba
    lbl = ''
    #if iz == 0: lbl = r'dlog($P$)/dlog($b_1$)'
    if iz == 0: lbl = r'$\theta=b_1$'
    ax[1].plot(kk, deriv, 'C%d--'%iz, label=lbl, lw=2)
    ax[1].text(0.1,3.5,"$\ell=0$", fontdict=font)
    
      
    #l=2
    p1 = np.loadtxt(dpathup + 'matchbias_mcut/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpathdn + 'matchbias_mcut/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    fid = np.loadtxt(dpath + 'HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    kk = fid[0]
    pk1 , pk2, pkf = p1[2], p2[2], fid[2]
    deriv = (pk1-pk2)/diffs['sigma']*fidp['sigma']/pkf
    ax[2].plot(kk, deriv, 'C%d-'%iz, lw=2)

    p1 = np.loadtxt(dpath + 'mcut_vp05/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    p2 = np.loadtxt(dpath + 'mcut_vm05/HI_pks_ll_{:06.4f}.txt'.format(aa)).T
    pk1 , pk2 = p1[2], p2[2]
    deriv = (pk1-pk2)/diffs['mcut']*fidp['mcut']/pkf
    deriv /= dbba
    ax[2].plot(kk, deriv, 'C%d--'%iz, lw=2)
    ax[2].text(0.1,3.5,"$\ell=2$", fontdict=font)

   
    

for axis in ax:
#     axis.set_xscale('log')
    axis.set_ylim(0.8, 4)
    axis.set_xlim(1e-2, 1.2)
    axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    axis.set_xlabel('k (h/Mpc)', fontsize=14)
    axis.axhline(2, color='gray', lw=0.7)
    axis.legend(prop=fontmanage, ncol=1, loc=3)
ax[2].set_ylim(0, 4)
#ax[0].set_ylabel(r'$\frac{{\rm dlog}(P(k))}{{\rm dlog}(q)}$', fontdict=font)
ax[0].set_ylabel(r'${\rm dlog}(P(k))\, /\, {\rm dlog}(\theta)$', fontdict=font)

for axis in ax.flatten():
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontproperties(fontmanage)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontproperties(fontmanage)

plt.tight_layout()
plt.savefig(figpath + 'deriv.pdf')
# plt.ylim(-1, 7)
