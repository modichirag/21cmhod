#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.signal import savgol_filter
#



suff = 'm1_00p3mh-alpha-0p8-subvol'
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass



def make_satdist_plot(fname, fsize=12):
    """Plot fraction of HI in satellites as function of halo mass"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,ax = plt.subplots(3,3,figsize=(13, 13))

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[1:,:]
            #nn = dist[:, 1] + 1
            #xx = dist[:, 0]/nn
            #satfrac = dist[:, 4]/nn/(dist[:, 2]/nn + 1e-10)
            xx = dist[:, 0]
            satfrac = dist[:, 4]/(dist[:, 2] + 1e-10)
            axis.plot(xx, satfrac, 'C%d.'%im, label=model)
            #cenfrac = dist[:, 3]/nn/(dist[:, 2]/nn + 1e-10)
            #axis.plot(xx, cenfrac, 'C%d.'%im, label=model)
            
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
    for axis in ax[-1]: axis.set_xlabel(r'M$(M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'HI Satellite fraction', fontsize=fsize)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #



def make_HIfrac_dh_plot(fname, fsize=12):
    """Plot HIfraction of total in given mass bin"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,ax = plt.subplots(3,3,figsize=(13, 13))

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[1:,:]
            xx = dist[:, 0]
            nn = dist[:, 1]
            h1frac = (dist[:, 2]*nn)/(dist[:, 2]*nn).sum()
            axis.plot(xx, h1frac, 'C%d.'%im, label=model)
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
    for axis in ax[-1]: axis.set_xlabel(r'M$(M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'HI fraction in bin', fontsize=fsize)
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

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[1:,:]
            nn = dist[:, 1]+1 
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
    for axis in ax[-1]: axis.set_xlabel(r'M$(M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'N halos', fontsize=fsize) 
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


def make_H1mh_plot(fname, fsize=13):
    """Plot mHI-mHalo relation for 2 models"""
    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.

    fig,ax = plt.subplots(3,3,figsize=(13, 13))

    for im, model in enumerate(['ModelA', 'ModelB']):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        print(model)
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            axis = ax.flatten()[iz]
            aa  = 1.0/(1.0+zz)
            dist = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa))[1:,:]
            xx = dist[:, 0]
            yy = dist[:, 2]
            axis.plot(xx, yy, 'C%do'%im, label=model)

            #Formatting
            axis.set_title('z = %0.1f'%zz, fontsize=fsize)
            axis.set_xscale('log')
            axis.set_yscale('log')
            #axis.set_ylim(0, 1.1)
            axis.grid(which='both')
            axis.grid(which='both')
            if iz == 0: axis.legend(fontsize=fsize)
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(fsize)
    # Put on some more labels.
    for axis in ax[-1]: axis.set_xlabel(r'M$(M_{\odot}/h)$', fontsize=fsize)
    for axis in ax[:, 0]: axis.set_ylabel(r'M$_{HI}(M_{\odot}/h)$', fontsize=fsize)
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


if __name__=="__main__":
    make_satdist_plot(figpath + 'HI_sat_fraction.pdf')
    make_HIfrac_dh_plot(figpath + 'HIfrac_dhalo.pdf')
    make_hmf_plot(figpath + 'HMF.pdf')
    make_H1mh_plot(figpath + 'HI_Mh.pdf')
    #
