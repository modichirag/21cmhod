#!/usr/bin/env python3
#
# Plots OmHI vs. z, with observational data.
#
import numpy as np
import matplotlib.pyplot as plt
from   scipy.interpolate import InterpolatedUnivariateSpline as Spline

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


def make_calib_plot():
    """Does the work of making the calibration figure."""

    zlist = [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]

    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(7,3.5))

    # The left hand panel is DLA bias vs. redshift.
    bDLA = np.loadtxt("../../data/boss_bDLA.txt")
    ax[0].errorbar(bDLA[:,0],bDLA[:,1],yerr=bDLA[:,2],fmt='s',color='m', mfc='None')
    ax[0].fill_between([1.5,3.5],[1.99-0.11,1.99-0.11],\
                                 [1.99+0.11,1.99+0.11],\
                       color='lightgrey',alpha=0.5)
    # The N-body results.
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        bbs, omz = [], []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            bias = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:10, 1].mean()
            bbs.append(bias)


        ax[0].plot(zlist[:3], bbs[:3],'C%do'%im, markersize=4, label=model, alpha=0.75)

        ss = Spline(zlist, bbs)
        ax[0].plot(np.linspace(1.5,3.5,100),ss(np.linspace(1.5,3.5,100)),'C%d--'%im)

    ################################################################
    # The right hand panel is OmegaHI vs. z.
    # Read in the data and convert to "normal" OmegaHI convention.
    dd = np.loadtxt("../../data/omega_HI_obs.txt")
    Ez = np.sqrt( 0.3*(1+dd[:,0])**3+0.7 )
    ax[1].errorbar(dd[:,0],1e-3*dd[:,1]/Ez**2,yerr=1e-3*dd[:,2]/Ez**2,\
                   fmt='s',mfc='None', color='m')
    # Plot the fit line.
    zz = np.linspace(0,7,100)
    Ez = np.sqrt( 0.3*(1+zz)**3+0.7 )
    ax[1].plot(zz,4e-4*(1+zz)**0.6/Ez**2,'k-')

    #data
    # The N-body results.
    for im, model in enumerate(models):
        dpath = '../../data/outputs/%s/%s/'%(suff, model)
        bbs, omz = [], []
        for iz, zz in enumerate(zlist):
            # Read the data from file.
            aa  = 1.0/(1.0+zz)
            omHI = np.loadtxt(dpath + "HI_dist_{:06.4f}.txt".format(aa)).T
            omHI = (omHI[1]*omHI[2]).sum()/bs**3/27.754e10 *(1+zz)**3
            omHI /= np.sqrt( 0.3*aa**-3+0.7 )**2
            omz.append(omHI)

        ax[1].plot(zlist, omz,'C%do'%im, markersize=4, alpha=0.75)
        ss = Spline(zlist, omz)
        ax[1].plot(np.linspace(2,6,100),ss(np.linspace(2,6,100)),'C%d--'%im)

    ###################################################################
    # Tidy up the plot.
    #
    ax[0].set_ylabel(r'b$_{\rm DLA}$(z)', fontdict=font)
    ax[1].set_ylabel(r'$\Omega_{\rm HI}$', fontdict=font)
    ax[0].set_xlim(1.9,3.5)
    ax[0].set_ylim(1,3)
    ax[1].set_xlim(1,6.25)
    ax[1].set_ylim(4e-6,3e-4)
    ax[1].set_yscale('log')
    ax[0].legend(prop=fontmanage)
    for axis in ax:
        axis.set_xlabel(r'z', fontdict=font)
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)


    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'calib.pdf')
    #





if __name__=="__main__":
    make_calib_plot()
    #
