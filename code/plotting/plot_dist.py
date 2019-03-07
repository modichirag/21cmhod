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
from matplotlib import rcParams
rcParams['font.family'] = 'serif'


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


##def make_omHI_plot(fname, fsize=12):
##    """Does the work of making the distribution figure."""
##    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
##    clist = ['b','c','g','m','r']
##    # Now make the figure.
##
##    fig,axis = plt.subplots(figsize=(6, 5))
##
##   # Read in the data and convert to "normal" OmegaHI convention.
##    dd = np.loadtxt("../../data/omega_HI_obs.txt")
##    Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
##    axis.errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
##                fmt='s',mfc='None')
##    # Plot the fit line.
##    zz = np.linspace(0,7,100)
##    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
##    axis.plot(zz,4e-4*(1+zz)**0.6/Ez**2,'k-')
##
##    #for im, model in enumerate(['ModelA', 'ModelB']):
##    for im, model in enumerate(models):
##        dpath = '../../data/outputs/%s/%s/'%(suff, model)
##        print(model)
##
##        omHI = np.loadtxt(dpath + "OmHI.txt")
##        #omHI[:, 1] /= 10
##        axis.plot(omHI[:, 0], omHI[:, 1], 'C%do'%im, label=model)
##
##        ss = ius(omHI[::-1, 0], omHI[::-1, 1])
##        axis.plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'C%d'%im)
##
##    axis.set_yscale('log')
##    axis.legend(fontsize=fsize)
##    for tick in axis.xaxis.get_major_ticks():
##        tick.label.set_fontsize(fsize)
##    for tick in axis.yaxis.get_major_ticks():
##        tick.label.set_fontsize(fsize)
##            
##    # Put on some more labels.
##    axis.set_xlabel(r'$z$')
##    axis.set_ylabel(r'$\Omega_{HI}$')
##    # and finish up.
##    plt.tight_layout()
##    plt.savefig(fname)
##    #
##
##
##




def make_omHI_plot(fname, fsize=12):
    """Does the work of making the distribution figure."""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,axis = plt.subplots(figsize=(6, 5))

   # Read in the data and convert to "normal" OmegaHI convention.
    dd = np.loadtxt("../../data/omega_HI_obs.txt")
    #Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
    #axis.errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
    #            fmt='s',mfc='None')
    axis.errorbar(dd[:,0],1e-3*dd[:,1],yerr=1e-3*dd[:,2],fmt='s',mfc='None', color='m')
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    axis.plot(zz,4e-4*(1+zz)**0.6,'k-')

    #for im, model in enumerate(['ModelA', 'ModelB', 'ModelC']):
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        omz = []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            omHI = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa)).T
            omHI = (omHI[1]*omHI[2]).sum()/bs**3/27.754e10
            omHI *= (1+zz)**3
            if iz == 0: axis.plot(zz, omHI, 'C%do'%im, label=model)
            else: axis.plot(zz, omHI, 'C%do'%im)
            omz.append(omHI)

        ss = ius(zlist, omz)
        axis.plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'C%d'%im)

    axis.set_yscale('log')
    axis.legend(fontsize=fsize)
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    axis.set_xlabel(r'$z$')
    axis.set_ylabel(r'$\Omega_{HI}$')
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


def make_satdist_plot(fname, fsize=12):
    """Plot fraction of HI in satellites as function of halo mass"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    fig,ax = plt.subplots(3,3,figsize=(13, 13), sharex=True, sharey=True)
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]
    fig,ax = plt.subplots(2,3,figsize=(13, 9), sharex=True, sharey=True)
    clist = ['b','c','g','m','r']
    # Now make the figure.


    for im, model in enumerate(['ModelA', 'ModelB']):
    #for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
            dist = dist[dist[:,1] !=0]
            xx = dist[:, 0]
            satfrac = dist[:, 4]/(dist[:, 2] + 1e-10)
            axis.plot(xx, satfrac, 'C%d'%im, marker='.', label=model)
            
            #Formatting
            axis.set_title('z = %0.1f'%zz, fontsize=fsize)
            axis.set_xscale('log')
            axis.set_ylim(0, 1.1)
            axis.grid(which='both')
            if iz == 0: axis.legend(fontsize=fsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    for axis in ax[-1]: axis.set_xlabel(r'M$(\rm M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'$\rm\frac{HI_{satellite}}{HI_{halo}}$', fontsize=fsize+2)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



def make_HIfrac_dh_plot(fname, fsize=12):
    """Plot HIfraction of total in given mass bin"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    fig,ax = plt.subplots(3,3,figsize=(13, 13), sharex=True, sharey=True)
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]
    fig,ax = plt.subplots(2,3,figsize=(13, 9), sharex=True, sharey=True)
    clist = ['b','c','g','m','r']
    # Now make the figure.


    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
            dist = dist[dist[:,1] !=0]
            xx = dist[:, 0]
            nn = dist[:, 1]
            h1frac = (dist[:, 2]*nn)/(dist[:, 2]*nn).sum()
            axis.plot(xx, h1frac, 'C%d'%im, marker='.', label=model)
            #cenfrac = dist[:, 3]/nn/(dist[:, 2]/nn + 1e-10)
            #axis.plot(xx, cenfrac, 'C%d.'%im, label=model)
            
            #Formatting
            axis.set_title('z = %0.1f'%zz, fontsize=fsize)
            axis.set_xscale('log')
            #axis.set_ylim(0, 1.1)
            axis.grid(which='both')
            if iz == 0: axis.legend(fontsize=fsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            
    # Put on some more labels.
    for axis in ax[-1]: axis.set_xlabel(r'M$(\rm M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'$\frac{1}{\rm{HI}_{total}}\frac{\rm{dHI}}{\rm{dlogM}_h}$', fontsize=fsize)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #





def make_hmf_plot(fname, fsize=13):
    """Plot halo mass function as a check for the code"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,ax = plt.subplots(3,3,figsize=(13, 13))

    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
            dist = dist[dist[:,1] !=0]
            nn = dist[:, 1]
            xx = dist[:, 0]
            axis.plot(xx, nn, 'C%do'%im, label=model)
            axis.set_title('z = %0.1f'%zz, fontsize=fsize)
            
            #Formatting
            axis.set_xscale('log')
            axis.set_yscale('log')
            #axis.set_ylim(0, 1.1)
            axis.grid(which='both')
            if iz == 0: axis.legend(fontsize=fsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
    # Put on some more labels.
    for axis in ax[-1]: axis.set_xlabel(r'M$(\rm M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'N halos', fontsize=fsize) 
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


def make_H1mh_plot(fname, fsize=13):
    """Plot mHI-mHalo relation for 2 models"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    fig,ax = plt.subplots(3,3,figsize=(13, 13), sharex=True, sharey=True)
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]
    fig,ax = plt.subplots(2,3,figsize=(13, 9), sharex=True, sharey=True)
    clist = ['b','c','g','m','r']
    # Now make the figure.

    
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            #dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[1:-1,:]
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[:,:]
            dist = dist[dist[:,1] !=0]
            xx = dist[:, 0]
            yy = dist[:, 2]
            axis.plot(xx, yy, 'C%d'%im, marker='.', label=model)

            #Formatting
            axis.set_title('z = %0.1f'%zz, fontsize=fsize)
            axis.set_xscale('log')
            axis.set_yscale('log')
            axis.set_ylim(8e4, 1.1e11)
            axis.grid(which='both')
            axis.grid(which='both')
            if iz == 0: axis.legend(fontsize=fsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
    # Put on some more labels.
    for axis in ax[-1]: axis.set_xlabel(r'M$(\rm M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'M$\rm _{HI}(M_{\odot}/h)$', fontsize=fsize)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


if __name__=="__main__":
    make_satdist_plot(figpath + 'HI_sat_fraction.pdf')
    make_HIfrac_dh_plot(figpath + 'HIfrac_dhalo.pdf')
    make_hmf_plot(figpath + 'HMF.pdf')
    make_H1mh_plot(figpath + 'HI_Mh.pdf')
    make_omHI_plot(figpath + 'omHI.pdf')
    #
