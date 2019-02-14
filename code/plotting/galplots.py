import numpy as np
import matplotlib.pyplot as plt
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower

from scipy.interpolate import InterpolatedUnivariateSpline as ius

import os, sys
sys.path.append('../code/utils/')
import tools, dohod
from time import time
import hodanalytics as hodanals
from HImodels import ModelA

dpath = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
# dpath = '../data/'

bs, nc = 256, 256
pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
comm = pm.comm
rank = comm.rank

# sim = '/lowres/%d-9100-fixed'%256
sim = '/highres/%d-9100-fixed'%2560
aafiles = np.array([0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333])

#aafiles = aafiles[:1]
zzfiles = np.array([round(tools.atoz(aa), 2) for aa in aafiles])

##Satellites

hpos, hmass, hid, h1mass, h1size = tools.readinhalos(aafiles, sim, HI_hod=None)

#for suff in ['-m1_00p3mh-alpha-0p8-v2', '-m1_00p3mh-alpha-0p9-v2', '-m1_00p5mh-alpha-0p8-v2', '-m1_00p5mh-alpha-0p9-v2']:
m1fac = 0.03
alpha = -0.8
hodparams = [m1fac, alpha]

for suff in [ '-m1_%02dp%dmh-alpha-0p8-subvol'%(int(m1fac*10), (m1fac*100)%10)]:

    subf = 'fid3/'+  suff[1:]
    try: 
        os.makedirs('./figs/%s'%subf)
    except:pass


    print('\n%s\n'%suff)
    start = time()
    cpos, cmass, chid, ch1mass, csize = tools.readincentrals(aafiles, suff, sim, HI_hod=None)
    spos, smass, shid, sh1mass, ssize = tools.readinsatellites(aafiles, suff, sim, HI_hod=None) 


    for iz, zz in enumerate(zzfiles):

        aa = 1/(zz+1)
        model = ModelA(aa)
        start = time()
        h1mass[zz] = model.assignhalo(hmass[zz])
        print('For halos : ', time() - start)
        start = time()
        sh1mass[zz] = model.assignsat(smass[zz])        
        print('For sat : ', time() - start)
        print('Populating centrals for ', iz)
        start = time()
        ch1mass[zz] = model.assigncen(h1mass[zz], sh1mass[zz], shid[zz], csize[zz], comm) 
        print('For cen : ', time() - start)
    
    print('Time to read all catalogs : ', time()-start)
    


    print('Bin Halos')
    hbins, hcount, hm, hh1 = {}, {}, {}, {}
    dmesh, pkm = {}, {}
    start = time()
    for iz, zz in enumerate(zzfiles):
        print(zz, 'Time : ', time()-start)
        #measure power
        dmesh[zz] = BigFileMesh(dpath + sim + '/fastpm_%0.4f/'%aafiles[iz] + '/dmesh_N%04d'%nc, '1').paint()
        pk = FFTPower(dmesh[zz]/dmesh[zz].cmean(), mode='1d').power
        k, pkm[zz] = pk['k'], pk['power']

        hbins[zz] = np.logspace(np.log10(hmass[zz][-1])-0.01, np.log10(hmass[zz][0])-0.01)
        hcount[zz], hm[zz], hh1[zz] = [np.zeros_like(hbins[zz]) for i in range(3)]
        for i in range(hbins[zz].size-1):
            r1, r2 = np.where(hmass[zz]>hbins[zz][i])[0][-1], np.where(hmass[zz]>hbins[zz][i+1])[0][-1]
            hcount[zz][i] = (r1-r2)
            hm[zz][i] = (hmass[zz][r2:r1].sum())
            hh1[zz][i] = (h1mass[zz][r2:r1].sum())
            



    print('Bin centrals & satellites')
    start = time()
    cm, sm, ch1, sh1, h1tot, scount = {}, {}, {}, {}, {}, {}
    for zz in zzfiles:
        print(zz, 'Time : ', time()-start)
        cm[zz], sm[zz], ch1[zz], sh1[zz], h1tot[zz], scount[zz] = [np.zeros_like(hbins[zz]) for i in range(6)]
        for i in range(hbins[zz].size-1):
            r1, r2 = np.where(hmass[zz]>hbins[zz][i])[0][-1], np.where(hmass[zz]>hbins[zz][i+1])[0][-1]
            smask, cmask = (shid[zz] < r1) & (shid[zz] > r2), (chid[zz] < r1) & (chid[zz] > r2)
            sh1[zz][i] = sh1mass[zz][smask].sum()
            ch1[zz][i] = ch1mass[zz][cmask].sum()
            sm[zz][i] = smass[zz][smask].sum()
            cm[zz][i] = cmass[zz][cmask].sum()
            scount[zz][i] = smask.sum()
            h1tot[zz][i] = (ch1[zz][i]+sh1[zz][i])


    ## Plot HOD/occupancy of satellites
    fig = plt.figure()
    for zz in zzfiles:
        plt.plot(hm[zz]/hcount[zz], scount[zz]/hcount[zz], label=zz)
    plt.xscale('log')
    plt.grid(which='both', lw=0.3)
    plt.xlabel('M_h (M$_\odot$/h)')
    plt.ylabel('Number of satellites')
    plt.legend()
    plt.savefig('./figs/%s/satcount%s'%(subf, suff))

    ## Plot mass of HI in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        axis.plot(hm[zz]/hcount[zz], hh1[zz]/hcount[zz], ':', lw=3, label='In halo')
        axis.plot(hm[zz]/hcount[zz], h1tot[zz]/hcount[zz], '--', lw=2, label='Central+satellite')
        axis.plot(hm[zz]/hcount[zz], ch1[zz]/hcount[zz], label='In centrals')
        axis.plot(hm[zz]/hcount[zz], sh1[zz]/hcount[zz], label='In satellites')
        axis.loglog()
        axis.set_title('z = %0.2f'%zz)
        axis.grid(which='both', lw=0.3)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{HI}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/%s/HIdist%s'%(subf, suff))


    ## Plot fraction of HI in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        #axis.plot(hm[zz]/hcount[zz], hh1[zz]/hcount[zz], '--', label='In halo')
        axis.plot(hm[zz]/hcount[zz], ch1[zz]/hh1[zz], label='In centrals of halo')
        axis.plot(hm[zz]/hcount[zz], sh1[zz]/hh1[zz], label='In satellites of halo')
        axis.plot(hm[zz]/hcount[zz], ch1[zz]/h1tot[zz], '--', lw=2, label='In centrals of total')
        axis.plot(hm[zz]/hcount[zz], sh1[zz]/h1tot[zz], '--', lw=2, label='In satellites of total')
        axis.set_xscale('log')
        axis.set_title('z = %0.2f'%zz)
        axis.grid(which='both', lw=0.3)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{HI}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/%s/HIdistratio%s'%(subf, suff))



    ## Plot mass  in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        axis.plot(hm[zz]/hcount[zz], hm[zz]/hcount[zz], ':', lw=3, label='In halo')
        axis.plot(hm[zz]/hcount[zz], cm[zz]/hcount[zz], label='In centrals')
        axis.plot(hm[zz]/hcount[zz], sm[zz]/hcount[zz], label='In satellites')
        axis.loglog()
        axis.set_title('z = %0.2f'%zz)
        axis.grid(which='both', lw=0.3)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{DM}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/%s/massdist%s'%(subf, suff))


    ## Plot fraction of mass in centrals and satellites
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    for iz, zz in enumerate(zzfiles):
        axis=ax.flatten()[iz]
        #axis.plot(hm[zz]/hcount[zz], hh1[zz]/hcount[zz], '--', label='In halo')
        axis.plot(hm[zz]/hcount[zz], cm[zz]/hm[zz], label='In centrals of halo')
        axis.plot(hm[zz]/hcount[zz], sm[zz]/hm[zz], label='In satellites of halo')
        axis.set_xscale('log')
        axis.set_title('z = %0.2f'%zz)
        axis.grid(which='both', lw=0.3)
    for axis in ax[:, 0]:axis.set_ylabel('M$_{DM}$')
    for axis in ax[-1, :]:axis.set_xlabel('M$_{h}$')
    ax[0, 0].legend()
    plt.savefig('./figs/%s/massdistratio%s'%(subf, suff))


    ## Plot mass function of centrals and satellites
    plt.figure(figsize=(8, 6))
    for i, zz  in enumerate(zzfiles):
        plt.hist(np.log10(smass[zz]), color='C%d'%i, bins=20, histtype='step', label=zz, log=True, lw=1.5, normed=False)
        plt.hist(np.log10(cmass[zz]), color='C%d'%i, bins=20, histtype='step', log=True, lw=2, normed=False, ls="--")
        plt.axvline(np.log10(dohod.HI_mass(1, aafiles[i], 'mcut')), color='C%d'%i, ls=":")
    plt.legend(ncol=2, loc=1)
    plt.xlim(8.5, 12)
    plt.ylim(10, 2e7)
    plt.savefig('./figs/%s/massfunc%s'%(subf, suff))



    ## Plot bias
    start = time()
    fig, ax = plt.subplots(1, 2, figsize = (9, 4))
    for iz, zz  in enumerate(zzfiles):
        sh1mesh = pm.paint(spos[zz], mass=sh1mass[zz])
        ch1mesh = pm.paint(cpos[zz], mass=ch1mass[zz])
        h1mesh = sh1mesh + ch1mesh
        pkh1 = FFTPower(h1mesh/h1mesh.cmean(), mode='1d').power['power']
        pkh1m = FFTPower(h1mesh/h1mesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
        b1h1 = pkh1m/pkm[zz]
        b1h1sq = pkh1/pkm[zz]

        ax[0].plot(k, b1h1, 'C%d'%iz, lw=1.5)
        ax[0].plot(k, b1h1sq**0.5, 'C%d--'%iz, lw=2)
        ax[1].plot(zz, b1h1[1:5].mean(), 'C%do'%iz, label='%0.2f'%(zz))
        ax[1].plot(zz, (b1h1sq**0.5)[1:5].mean(), 'C%d*'%iz)

    for axis in ax: 
        axis.legend()
        ax[0].set_xscale('log')
        axis.set_ylim(1, 5)
        axis.grid(which='both', lw=0.3)
    fig.savefig('./figs/%s/bias%s.png'%(subf, suff))

    print('Time to estimate bias : ', time()-start)

