#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
#
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

print(font)


#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='which box size simulation', default='small')
args = parser.parse_args()
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256

figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass


models = ['ModelA', 'ModelB', 'ModelC']





def make_bias_omHI_plot(fname, fsize=10):
    """Does the work of making the distribution figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]

    # Now make the figure.

    fig, axar = plt.subplots(1, 2, figsize=(7, 3.5), sharex=True)

    #plot bias observation
    bDLA = np.loadtxt("../../data/boss_bDLA.txt")
    axar[1].errorbar(bDLA[:,0],bDLA[:,1],yerr=bDLA[:,2],fmt='s', label='BOSS', color='m', mfc='None')
    axar[1].fill_between([1.5,3.5],[1.99-0.11,1.99-0.11],\
                                 [1.99+0.11,1.99+0.11],\
                       color='lightgrey',alpha=0.5)

    #plot omHI observation
    dd = np.loadtxt("../../data/omega_HI_obs.txt")
    #axar[1].errorbar(dd[:,0],1e-3*dd[:,1],yerr=1e-3*dd[:,2],fmt='s',mfc='None', color='m', label="Crighton '15")
    axar[0].errorbar(dd[:,0],dd[:,1],yerr=dd[:,2],fmt='s',mfc='None', color='m', label="Crighton '15")
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    axar[0].plot(zz,4e-4*(1+zz)**0.6,'k-')

    #Do for models
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)

        bbs, omz = [], []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            bias = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:10, 1].mean()
            bbs.append(bias)
            omHI = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa)).T
            omHI = (omHI[1]*omHI[2]).sum()/bs**3/27.754e10 *(1+zz)**3 *1e3
            if im == 1: omHI*(1+3)**3/(1+zz)**3
            omz.append(omHI)

        axar[1].plot(zlist, bbs, 'C%d'%im, marker='.', label=model)
        axar[0].plot(zlist, omz, 'C%d'%im, marker='.')

    axar[1].set_ylabel(r'b$_{\rm DLA}$(z)', fontdict=font)
    axar[0].set_ylabel(r'$\Omega_{\rm HI} \times 10^{-3}$', fontdict=font)
    #axar[1].set_yscale('log')
    axar[0].set_ylim(0.5, 1.5)
    for axis in axar:
        axis.set_xlim(1.5, 6.1)
        axis.legend(prop=fontmanage)
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

        axis.set_xlabel(r'z', fontdict=font)

    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


def make_HIdist_plot(fname, fsize=13):
    """Plot fraction of HI in satellites as function of halo mass"""
    zlist = [2.0,4.0,6.0]
    fig, axar = plt.subplots(3,3,figsize=(9, 6), sharex=True, sharey='row')

    # Now make the figure.
    #for im, model in enumerate(['ModelA', 'ModelB']):
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axar[0, iz].set_title('z = %0.1f'%zz, fontdict=font)
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
            dist = dist[dist[:,1] !=0]

            xx = np.log10(dist[:, 0])
            if im == 0 or im == 2 : dist[: ,2:] *= (1+zz)**3
            else: dist[: ,2:] *= (1+3)**3
            axar[0, iz].plot(xx, dist[:, 2], 'C%d'%im, marker='.', label=model, lw=2)
            
            
            nn = dist[:, 1]
            difflogm = np.diff(np.log(xx))[0]
            print(difflogm)
            h1frac = (dist[:, 2]*nn)/(dist[:, 2]*nn).sum()
            h1frac /= difflogm
            axar[1, iz].plot(xx, h1frac, 'C%d'%im, marker='.', lw=2)

            satfrac = dist[:, 4]/(dist[:, 2] + 1e-10)
            axar[2, iz].plot(xx, satfrac, 'C%d'%im, marker='.', lw=2)       

    axar[0, 0].legend(prop=fontmanage)


    for axis in axar.flatten():
        #axis.set_xscale('log')
        axis.grid(which='both')
        axis.grid(which='both')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            #tick.label.set_fontsize(fsize-1)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            #tick.label.set_fontsize(fsize-1)

    #for axis in axar[-1]: axis.set_xlabel(r'M$\rm_h$$(\rm M_{\odot}/h)$', fontdict=font)
    for axis in axar[-1]: axis.set_xlabel(r'log$_{\rm 10}$(M$\rm_h$$(\rm M_{\odot}/h) )$', fontdict=font)

    #yaxis
    for axis in axar[0, :]: 
        axis.set_yscale('log')
        axis.set_ylim(8e4, 5e12)
    axar[0, 0].set_ylabel(r'M$\rm _{HI}(M_{\odot}/h)$', fontdict=font)

    #for axis in axar[1, :]: 
        #axis.set_ylim(0, 1.1)
    #axar[1, 0].set_ylabel(r'$\frac{1}{\rm{HI}_{total}}\frac{\rm{dHI}}{\rm{dlogM}_h}$', fontdict=font)
    axar[1, 0].set_ylabel(r'$d\,{\rm f_{HI}}\,/\,d\,{\rmlog\, M_h}$', fontdict=font)
    for axis in axar[2, :]: 
        axis.set_ylim(-0.05, 1.02)

    #axar[2, 0].set_ylabel(r'$\rm\frac{HI_{satellite}}{HI_{halo}}$', fontdict=font)
    #axar[2, 0].set_ylabel(r'HI$_{\rm sat}$/HI$_{\rm halo}$', fontdict=font)
    axar[2, 0].set_ylabel(r'M$\rm _{HI}^{sat}\,/\,M _{HI}^{h}$', fontdict=font)

    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #
            




if __name__=="__main__":
    make_HIdist_plot(figpath + 'HI_dist_comparev2.pdf')
    make_bias_omHI_plot(figpath + 'bias_omHI_comparev2.pdf')
    #
