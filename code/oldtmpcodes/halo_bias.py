import numpy as np
import matplotlib.pyplot as plt
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower

import os, sys
sys.path.append('../code/utils/')
import tools, dohod
from time import time


dpath = '/project/projectdirs/m3127/H1mass/'
myscratch = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
# dpath = '../data/'

bs, nc = 256, 256
pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
# sim = '/lowres/%d-9100-fixed'%256
sim = '/highres/%d-9100-fixed'%2560
aafiles = np.array([0.1429, 0.1538, 0.1667, 0.1818, 0.2000, 0.2222, 0.2500, 0.2857, 0.3333])

#aafiles = aafiles[:1]
zzfiles = np.array([round(tools.atoz(aa), 2) for aa in aafiles])


hpos, hmass, hid, h1mass = tools.readinhalos(aafiles, sim)

print('Bin Halos')
hbins, hcount, hm, hh1 = {}, {}, {}, {}
dmesh, pkm = {}, {}
for iz, zz in enumerate(zzfiles):
    
    print('For redshift : ', zz)
    #measure power
    dmesh[zz] = BigFileMesh(dpath + sim + '/fastpm_%0.4f/'%aafiles[iz] + '/dmesh_N%04d'%nc, '1').paint()
    pk = FFTPower(dmesh[zz]/dmesh[zz].cmean(), mode='1d').power
    k, pkm = pk['k'], pk['power']

    hbins[zz] = np.logspace(np.log10(hmass[zz][-1])-0.01, np.log10(hmass[zz][0])-0.01, 100)

    bias_table = []
    massval = []
    inb = 6

    posmesh = pm.paint(hpos[zz])
    massmesh = pm.paint(hpos[zz], mass=hmass[zz])

    pkhm = FFTPower(massmesh/massmesh.cmean(), mode='1d').power['power']
    pkhmx = FFTPower(massmesh/massmesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
    pkhp = FFTPower(posmesh/posmesh.cmean(), mode='1d').power['power']
    pkhpx = FFTPower(posmesh/posmesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
    biases = [(pkhm[1:inb]/pkm[1:inb]).mean()**0.5, (pkhp[1:inb]/pkm[1:inb]).mean()**0.5, (pkhmx[1:inb]/pkm[1:inb]).mean(), (pkhpx[1:inb]/pkm[1:inb]).mean()]
    print(biases)

##    for i in range(hbins[zz].size-1):
##        if i %10 == 0 : print(i)
##        r1, r2 = np.where(hmass[zz]>hbins[zz][i])[0][-1], np.where(hmass[zz]>hbins[zz][i+1])[0][-1]
##        if r1 - r2 == 0:
##            bias_table.append([0, 0, 0, 0])
##            massval.append([hbins[zz][i], 0, 0])
##        else:
##            massval.append((hbins[zz][i], hmass[zz][r2:r1].mean(), r1-r2))
##            posmesh = pm.paint(hpos[zz][r2:r1])
##            massmesh = pm.paint(hpos[zz][r2:r1], mass=hmass[zz][r2:r1])
##
##            pkhm = FFTPower(massmesh/massmesh.cmean(), mode='1d').power['power']
##            pkhmx = FFTPower(massmesh/massmesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
##            pkhp = FFTPower(posmesh/posmesh.cmean(), mode='1d').power['power']
##            pkhpx = FFTPower(posmesh/posmesh.cmean(), second=dmesh[zz]/dmesh[zz].cmean(), mode='1d').power['power']
##            biases = [(pkhm[1:inb]/pkm[1:inb]).mean()**0.5, (pkhp[1:inb]/pkm[1:inb]).mean()**0.5, (pkhmx[1:inb]/pkm[1:inb]).mean(), (pkhpx[1:inb]/pkm[1:inb]).mean()]
##            bias_table.append(biases)
##
##    bias_table = np.array(bias_table)
##    massval = np.array(massval)
##    np.savetxt(dpath + sim + '/fastpm_%0.4f/'%aafiles[iz] + '/halobias_bins.txt', np.concatenate((massval, bias_table), axis=1).real, \
##               header='min_mass, mean_mass, mcount, auto_mass, auto_pos, cross_mass, cross_pos', fmt='%0.06e')
##    
