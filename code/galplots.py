import numpy as np
import matplotlib.pyplot as plt
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower

import os, sys
sys.path.append('../code/utils/')
import tools, dohod
from time import time


dpath = '/project/projectdirs/m3127/H1mass/'
# dpath = '../data/'

bs, nc = 256, 256
pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
# sim = '/lowres/%d-9100-fixed'%256
sim = '/highres/%d-9100-fixed'%2560
aafiles = [0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333]
#aafiles = aafiles[:2]
zzfiles = [round(tools.atoz(aa), 2) for aa in aafiles]


##Halos and centrals

hpos, hmass, h1mass = {}, {}, {}
cpos, cmass, ch1mass, chid = {}, {}, {}, {}


print('Read Halos and Centrals')
for i, aa in enumerate(aafiles):
    zz = zzfiles[i]
    print(zz)
    start = time()
    halos = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/halocat/'%aa)
    cen = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/cencat/'%aa)
    hmass[zz], h1mass[zz] = halos["Mass"].compute(), halos['H1mass'].compute()
    cmass[zz], ch1mass[zz] = cen["Mass"].compute(), cen['H1mass'].compute()
    hpos[zz], cpos[zz] = halos['Position'].compute(), cen['Position'].compute()
    chid[zz] = cen['HaloID'].compute()
    print('Time ', time()-start)


print('Bin Centrals')
hbins, hcount, hm, ch1 = {}, {}, {}, {}
dmesh, ch1mesh, pkm, pkch1, pkmch1x = {}, {}, {}, {}, {}
for iz, zz in enumerate(zzfiles):
    print(zz)
    #measure power
    dmesh[zz] = BigFileMesh(dpath + sim + '/fastpm_%0.4f/'%aafiles[iz] + '/dmesh_N%04d'%nc, '1').paint()
    pk = FFTPower(dmesh[zz]/dmesh[zz].cmean(), mode='1d').power
    k, pkm[zz] = pk['k'], pk['power']
    ch1mesh[zz] = pm.paint(cpos[zz], mass=ch1mass[zz])    
    #pkch1[zz] = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
    #pkmch1x[zz] = FFTPower(h1mesh/h1mesh.cmean(), second=dm/dm.cmean(), mode='1d').power['power']

    hbins[zz] = np.logspace(np.log10(hmass[zz][-1])-0.01, np.log10(hmass[zz][0])-0.01)
    hcount[zz], hm[zz], ch1[zz] = [np.zeros_like(hbins[zz]) for i in range(3)]
    for i in range(hbins[zz].size-1):
        r1, r2 = np.where(hmass[zz]>hbins[zz][i])[0][-1], np.where(hmass[zz]>hbins[zz][i+1])[0][-1]
        hcount[zz][i] = (r1-r2)
        hm[zz][i] = (hmass[zz][r2:r1].sum())
        ch1[zz][i] = (ch1mass[zz][(chid[zz] < r1) & (chid[zz] > r2)].sum())



##Satellites

#for suff in ['_b3p0', '_b2p5', '_b3p5', '_b4p0']:
for suff in ['-min_1p0h1-m1_10p0h1',  '-min_1p0h1-m1_100p0h1', '-min_1p0h1-m1_50p0h1', '-min_0p5h1-m1_10p0h1']:
    print(suff)
    spos, smass, sh1mass, shid = {}, {}, {}, {}

    print('Read satellites')
    for i, aa in enumerate(aafiles):
        zz = zzfiles[i]
        print(zz)
        start = time()
        sat = BigFileCatalog(dpath + sim+ '/fastpm_%0.4f/satcat'%aa+suff)
        smass[zz], sh1mass[zz] = sat["Mass"].compute(), sat['H1mass'].compute()
        spos[zz] = sat['Position'].compute()
        shid[zz] = sat['HaloID'].compute()
        print(time()-start)


    print('Bin satellites')
    sh1, hh1, scount = {}, {}, {}
    for zz in zzfiles:
        print(zz)
        sh1[zz], hh1[zz], scount[zz] = [np.zeros_like(hbins[zz]) for i in range(3)]
        for i in range(hbins[zz].size-1):
            r1, r2 = np.where(hmass[zz]>hbins[zz][i])[0][-1], np.where(hmass[zz]>hbins[zz][i+1])[0][-1]
            sh1[zz][i] = (sh1mass[zz][(shid[zz] < r1) & (shid[zz] > r2)].sum())
            scount[zz][i] = (((shid[zz] < r1) & (shid[zz] > r2)).sum())
            hh1[zz][i] = (ch1[zz][i]+sh1[zz][i])


    ## Plot HOD/occupancy of satellites
    fig = plt.figure()
    for zz in zzfiles:
        plt.plot(hm[zz]/hcount[zz], scount[zz]/hcount[zz], label=zz)
    plt.xscale('log')
    plt.xlabel('M_h (M$_\odot$/h)')
    plt.ylabel('Number of satellites')
    plt.legend()
    plt.savefig('./figs/satcount%s'%suff)

    ## Plot mass of HI in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        axis.plot(hm[zz]/hcount[zz], hh1[zz]/hcount[zz], '--', lw=2, label='In halo')
        axis.plot(hm[zz]/hcount[zz], ch1[zz]/hcount[zz], label='In centrals')
        axis.plot(hm[zz]/hcount[zz], sh1[zz]/hcount[zz], label='In satellites')
        axis.loglog()
        axis.set_title('z = %0.2f'%zz)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{HI}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/HIdist%s'%suff)


    ## Plot fraction of HI in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        #axis.plot(hm[zz]/hcount[zz], hh1[zz]/hcount[zz], '--', label='In halo')
        axis.plot(hm[zz]/hcount[zz], ch1[zz]/hh1[zz], label='In centrals')
        axis.plot(hm[zz]/hcount[zz], sh1[zz]/hh1[zz], label='In satellites')
        axis.set_xscale('log')
        axis.set_title('z = %0.2f'%zz)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{HI}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/HIdistratio%s'%suff)


    ## Plot mass function of centrals and satellites
    plt.figure(figsize=(8, 6))
    for i, zz  in enumerate(zzfiles):
    #     plt.hist(np.log10(smass[zz]), color='C%d'%i, bins=50, histtype='step', label=zz, log=False, lw=2, normed=True)
    #     plt.hist(np.log10(cmass[zz]), color='C%d'%i, bins=50, histtype='step', label=zz, log=False, lw=2, normed=True, ls="--")
        plt.hist(np.log10(smass[zz]), color='C%d'%i, bins=20, histtype='step', label=zz, log=True, lw=1.5, normed=False)
        plt.hist(np.log10(cmass[zz]), color='C%d'%i, bins=20, histtype='step', log=True, lw=2, normed=False, ls="--")
        plt.axvline(np.log10(dohod.HI_mass(1, aafiles[i], 'mcut')), color='C%d'%i, ls=":")
    plt.legend(ncol=2, loc=1)
    plt.xlim(8.5, 12)
    plt.ylim(10, 2e7)
    plt.savefig('./figs/massfunc%s'%suff)



    ## Plot bias
    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
    for iz, zz  in enumerate(zzfiles):
        sh1mesh = pm.paint(spos[zz], mass=sh1mass[zz])
        h1mesh = sh1mesh + ch1mesh[zz]
        pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
        pkh1m = FFTPower(h1mesh/h1mesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
        b1h1 = pkh1m/pkm[zz]
        b1h1sq = pkh1/pkm[zz]
        #ofolder = dpath + '/%s/fastpm_%0.4f/'%(sim, aafiles[iz])
        #np.savetxt(ofolder+'bias%s.txt'%suff, np.stack((k, b1h1, b1h1sq**0.5), axis=1), 
        #                                       header='k, pkh1xm/pkm, pkh1/pkm^0.5')

        ax[0].plot(k, b1h1, 'C%d'%iz, lw=1.5)
        ax[0].plot(k, b1h1sq**0.5, 'C%d--'%iz, lw=2)
        ax[1].plot(zz, b1h1[1:5].mean(), 'C%do'%iz, label='%0.2f'%(zz))
        ax[1].plot(zz, (b1h1sq**0.5)[1:5].mean(), 'C%d*'%iz)

    for axis in ax: 
        axis.legend()
        ax[0].set_xscale('log')
        axis.set_ylim(1, 5)
        #axis.grid(which='both', lw=0.3)
    fig.savefig('./figs/bias%s.png'%suff)

