#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.signal import savgol_filter
#
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use')
args = parser.parse_args()
print(args, args.model)



bs = 1024
model = args.model #'ModelD'
suff = 'm1_00p3mh-alpha-0p8-subvol'
if bs == 1024: suff = suff + '-big'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass

svfilter = True
if bs == 256: winsize = 7
elif bs == 1024: winsize  = 19



def make_bao_plot(fname):
    """Does the work of making the BAO figure."""
    zlist = [2.0,2.5,3.0,4.0,5.0,6.0]
    zlist = [2.0,4.0,2.5,5.0,3.0,6.0]
    clist = ['b','c','g','m','r','y']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,3.0),sharey=True)
    ii,jj  = 0,0
    for zz,col in zip(zlist,clist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        #redshift space
        pks = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
        pks = np.loadtxt(dpath + "HI_pks_ll_{:06.4f}.txt".format(aa))[1:,:]
        pks = np.interp(pkd[:,0],pks[:,0],pks[:,1])
        
        # Now read linear theory and put it on the same grid -- currently
        # not accounting for finite bin width.
        lin = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        lin = np.interp(pkd[:,0],lin[:,0],lin[:,1])
        
        # Take out the broad band.
        if not svfilter: # Use smoothing spline as broad-band/no-wiggle.
            knots=np.arange(0.05,0.5,0.05)
            ps = pkd[:,2]**2 *pkd[:,3]
            ss  = Spline(pkd[:,0],ps,t=knots)
            rat = ps/ss(pkd[:,0])
            ss  = Spline(pkd[:,0],pks,t=knots)
            rats = pks/ss(pkd[:,0])
            ss  = Spline(pkd[:,0],lin,t=knots)
            ratlin = lin/ss(pkd[:,0])
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ps = pkd[:,2]**2 *pkd[:,3]
            ss  = savgol_filter(ps, winsize,polyorder=2)
            rat = ps/ss
            ss  = savgol_filter(pks, winsize ,polyorder=2)
            rats = pks/ss
            ss  = savgol_filter(lin,winsize,polyorder=2)
            ratlin = lin/ss

        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),col+'-',\
                    label="$z={:.1f}$".format(zz))
        ax[ii].plot(pkd[:,0],rats+0.2*(jj//2),col+'--', alpha=0.7, lw=2)
        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),'k:', lw=1)

        ii = (ii+1)%2
        jj =  jj+1
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5)
        ax[ii].set_xlim(0.03,0.4)
        ax[ii].set_ylim(0.75,1.5)
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[1].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$')
    ax[0].set_ylabel(r'$P(k)/P_{\rm nw}(k)$+offset')
    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


if __name__=="__main__":
    make_bao_plot(figpath + 'HI_bao_%s.pdf'%model)
    #
