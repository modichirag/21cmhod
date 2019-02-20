#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.signal import savgol_filter
#
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use')
args = parser.parse_args()
print(args, args.model)



model = args.model #'ModelD'
suff = 'm1_00p3mh-alpha-0p8-subvol'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass



def make_bao_plot(fname):
    """Does the work of making the BAO figure."""
    zlist = [2.0,3.0,4.0,5.0,6.0]
    clist = ['b','c','g','m','r']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(6,3.0),sharey=True)
    ii,jj  = 0,0
    for zz,col in zip(zlist,clist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]


        # Now read linear theory and put it on the same grid.
        lin = np.loadtxt("../../data/pklin_{:06.4f}.txt".format(aa))
        dk  = pkd[1,0]-pkd[0,0]
        kk  = np.linspace(pkd[0,0]-dk/2,pkd[-1,0]+dk/2,5000)
        tmp = np.interp(kk,lin[:,0],lin[:,1])
        lin = np.zeros_like(pkd)
        for i in range(pkd.shape[0]):
            lin[i,0] = pkd[i,0]
            ww       = np.nonzero( (kk>pkd[i,0]-dk/2)&(kk<pkd[i,0]+dk/2) )
            lin[i,1] = np.sum(kk[ww]**2*tmp[ww])/np.sum(kk[ww]**2)
        # Take out the broad band.
        if False: # Use smoothing spline as broad-band/no-wiggle.
            knots=np.arange(0.05,0.5,0.05)
            ss  = Spline(pkd[:,0],pkd[:,1],t=knots)
            rat = pkd[:,1]/ss(pkd[:,0])
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ss  = savgol_filter(pkd[:,1],7,polyorder=2)
            rat = pkd[:,1]/ss
        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),col+'-',\
                    label="$z={:.1f}$".format(zz))
        if False: # Use smoothing spline as broad-band/no-wiggle.
            ss  = Spline(pkd[:,0],lin,t=knots)
            rat = lin/ss(pkd[:,0])
        else:	# Use Savitsky-Golay filter for no-wiggle.
            ss  = savgol_filter(lin[:,1],7,polyorder=2)
            rat = lin[:,1]/ss
        ax[ii].plot(pkd[:,0],rat+0.2*(jj//2),col+':')
        ii = (ii+1)%2
        jj =  jj+1
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5)
        ax[ii].set_xlim(0.05,0.4)
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
