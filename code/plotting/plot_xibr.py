#!/usr/bin/env python3
#
# Plots the real-space correlation functions and biases for the HI.
#
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
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
model = 'ModelA'
dpath = '../../data/outputs/%s/%s/'%(suff, model)


def make_xib_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [2.0,4.0,6.0]
    blist = [1.8,2.6,3.5]
    clist = ['b','g','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(7,3.5))

    for zz,bb,col in zip(zlist,blist,clist):

        aa = 1.0/(1.0+zz)
        mfc = col
        # Put on a fake symbol for the legend.
        ax[0].plot([100],[100],'s',color=col,label="z={:.1f}".format(zz))
        # Plot the data, xi_mm, xi_hm, xi_hh

        xim = np.loadtxt(dpath + "ximatter_{:06.4f}.txt".format(aa))
        ax[0].plot(xim[:,0],xim[:,1],'d--',color=col,mfc=mfc,alpha=0.75, markersize=3)

        xix = np.loadtxt(dpath + "ximxh1mass_{:06.4f}.txt".format(aa))
        ax[0].plot(xix[:,0],xix[:,1],'s-',color=col,mfc='None',alpha=0.75, markersize=3)

        xih = np.loadtxt(dpath + "xih1mass_{:06.4f}.txt".format(aa))
        ax[0].plot(xih[:,0],xih[:,1],'o--',color=col,mfc=mfc,alpha=0.75, markersize=3)


        # and the inferred biases.
        ba = np.sqrt(ius(xih[:,0],xih[:,1])(xim[:,0])/xim[:,1])
        bx = ius(xix[:,0], xix[:,1])(xim[:,0])/xim[:,1] 
        #ba = np.sqrt(xih[:,1]/xim[:,1])
        #bx = xix[:,1]/xim[:,1]
        ax[1].plot(xih[:,0],bx,'s-' ,color=col,mfc='None',alpha=0.75, markersize=3)
        ax[1].plot(xih[:,0],ba,'o--',color=col,mfc=mfc,alpha=0.75, markersize=3)
        # put on a line for Sigma -- labels make this too crowded.
        pk  = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        Sig = np.sqrt(np.trapz(pk[:,1],x=pk[:,0])/6./np.pi**2)
        ax[0].plot([Sig,Sig],[1e-10,1e10],':',color='darkgrey')
        # Tidy up the plot.
        ax[0].set_ylim(0.005,20.)
        ax[0].set_yscale('log')
        #
        ax[1].set_ylim(1.0,5.0)

    ax[0].legend(prop=fontmanage)
    # Put on some more labels.
    ax[0].set_ylabel(r'$\xi(r)$', fontdict=font)
    ax[1].set_ylabel(r'Bias', fontdict=font)
    for axis in ax:
        axis.set_xlabel(r'$r\quad [h^{-1}\,{\rm Mpc}]$', fontdict=font)
        axis.set_xlim(0.7,30.0)
        axis.set_xscale('log')
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)

    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'HI_xib_%s.pdf'%model)
    #




if __name__=="__main__":
    make_xib_plot()
    #
