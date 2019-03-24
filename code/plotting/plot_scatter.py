#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from scipy.special import spherical_jn
j0 = lambda x: spherical_jn(0, x)
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
#parser.add_argument('-m', '--model', help='model name to use')
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
args = parser.parse_args()


model = 'ModelA' #args.model
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
dpathxi = '../../data/outputs/%s/%s/'%(suff, model)
if boxsize == 'big':
    suff = suff + '-big'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass


ncube = 8



def make_scatter_plot():
    """Does the work of making the real-space xi(r) and b(r) figure."""
    zlist = [5.0, 6.0]

    for profile in [2.0, 2.5, 2.9]:
        # Now make the figure.

        for iz, zz in enumerate(zlist):

            fig,axar = plt.subplots(2,3,figsize=(10,7))
            aa = 1.0/(1.0+zz)
            # Put on a fake symbol for the legend.


            vals = np.loadtxt(dpath +'scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, (profile*10)//10, (profile*10)%10, aa)).T
            for i in [3, 4, 5, 6, 7]: vals[i] /= vals[i].mean()
            dm, h1fid, h1new, lum, uv = vals[3:]

            def fit(x, y, axis, ic=0, xlab=None, ylab=None):
                mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
                mx = np.median(x)
                mask = (x>mx-2*sx) & (x<mx+2*sx)
                x, y = x[mask], y[mask]
                m, c = np.polyfit(x, y, deg=1)
                print(mx, sx)
                axis.plot(x, y, 'C%d.'%ic,ms=2)
                axis.axvline(mx, lw=0.5, color='gray')
                #axis.axvline(mx+1*sx)
                #axis.axvline(mx-1*sx)
                xx = np.linspace(x.min(), x.max())
                yy = xx*m + c
                axis.plot(xx, yy,'r', label='slope={:.2f}\nintercept={:.2f}'.format(m, c))
                axis.legend(ncol=1)
                axis.set_xlabel(xlab)
                axis.set_ylabel(ylab)
                #axis.set_ylim(m*xx.min()+c, m*xx.max()+c)
                return xx, m*xx+c, m, c

            fit(dm, h1fid, axar[0, 0], xlab='$\delta_m$', ylab=r'$\delta HI_{fid}$')
            fit(dm, h1new, axar[0, 1], xlab='$\delta_m$', ylab=r'$\delta HI_{new}$')
            fit(dm, uv, axar[0, 2], xlab='$\delta_m$', ylab='$\delta_{UV}$')

            fit(uv, h1fid, axar[1, 0], xlab='$\delta_{UV}$', ylab=r'$\delta HI_{fid}$')
            fit(uv, h1new, axar[1, 1], xlab='$\delta_{UV}$', ylab=r'$\delta HI_{new}$')
            fit(uv, h1fid-h1new, axar[1, 2], xlab='$\delta_{UV}$', ylab=r'$\delta HI_{fid}-\delta HI_{new}$')

        #
            # and finish up.
            plt.tight_layout()
            plt.savefig(figpath + 'scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.pdf'.format(ncube, (profile*10)//10, (profile*10)%10, aa))
            #



if __name__=="__main__":
    make_scatter_plot()
    #
