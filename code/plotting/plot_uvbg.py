#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from scipy.optimize import minimize

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps
from scipy.special import spherical_jn
j0 = lambda x: spherical_jn(0, x)
#

sys.path.append('../')
from uvbg import mfpathz

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
stellar = False



suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass


ncube = 8


def solvequad(a, b, c, plus=True):
    if plus: return (-b + (b**2 - 4*a*c)**0.5) / 2 /a
    else: return (-b - (b**2 - 4*a*c)**0.5) / 2 /a



def make_bk_scatter_plot():
    """"""
    zlist = [3.5, 4.0, 5.0, 6.0]
    #zlist = [5.0, 6.0]

    fig, axar = plt.subplots(len(zlist), 2, figsize=(10, len(zlist)*3), sharex='col', sharey='col')
    
    for iz, zz in enumerate(zlist):
        
        print('For redshift = {:.2f}'.format(zz))

        aa = 1.0/(1.0+zz)
        ax = axar[iz]
        # Put on a fake symbol for the legend.
        
        h1fid = np.loadtxt(dpath + '/HI_bias_{:.4f}.txt'.format(aa)).T
        lum = np.loadtxt(dpath + 'uvbg/Lum_bias_{:.4f}.txt'.format(aa)).T
        if stellar:
            uv = np.loadtxt(dpath + 'uvbg/UVbg_star_bias_{:.4f}.txt'.format(aa)).T
        else:
            uv = np.loadtxt(dpath + 'uvbg/UVbg_bias_{:.4f}.txt'.format(aa)).T
            
        #
        pkg = uv[1]**2*uv[3]
        pkmg = uv[1]*uv[3]
        pkh1 = h1fid[2]**2*h1fid[3]
        b1h1 = h1fid[1]
        
        for ip, profile in enumerate([2.0, 2.5, 2.9]):

            ap = 'ap{:1.0f}p{:1.0f}'.format((profile*10)//10, (profile*10)%10)
            if stellar:
                h1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_star_bias_{:6.4f}.txt'.format( ap, aa)).T
            else:
                h1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_bias_{:6.4f}.txt'.format( ap, aa)).T
            k = h1uv[0]
            pkh1g = h1uv[2]**2*h1uv[3]
            bb = np.zeros_like(pkg)
            for i in range(pkg.size):
                bb[i] = solvequad(pkg[i], 2*b1h1[i]*pkmg[i], pkh1[i]-pkh1g[i])
            #ax[0].plot(k, bb, label='profile = %.1f'%profile)
            ax[0].plot(k, bb)
            #bmean = bb[k<0.1][1:].mean()
            bmean = bb[1:6].mean()
            ax[0].axhline(bmean, color='C%d'%ip, lw=0.5, ls='--', label = 'b = %0.2f'%bmean)
            ax[0].axvline(k[6], color='gray', lw=0.5)


            try:
                if stellar:
                    vals = np.loadtxt(dpath +'/uvbg/scatter_star_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
                else:
                    vals = np.loadtxt(dpath +'/uvbg/scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
                for i in [3, 4, 5, 6, 7]: vals[i] /= vals[i].mean()
                dm, h1fid, h1new, lum, uv = vals[3:]

                def fit(x, y, axis, ic=ip, xlab=None, ylab=None):
                    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
                    #mask = (x>mx-3*sx) & (x<mx+3*sx)
                    #x, y = x[mask], y[mask]
                    m, c = np.polyfit(x, y, deg=1)
                    linearfit = HuberRegressor().fit(x.reshape(-1, 1), y)
                    m, c = linearfit.coef_[0], linearfit.intercept_

                    print(m, c)
                    axis.plot(x, y, 'C%d.'%ic,ms=2)
                    axis.axvline(mx, lw=0.5, color='gray')
                    xx = np.linspace(x.min(), x.max())
                    yy = xx*m + c
                    axis.plot(xx, yy, 'C%d'%ip, label='m={:.2f}'.format(m))
                    #axis.set_xlim(mx-2*sx, mx+2*sx)
                    #axis.set_ylim(m*(mx-2*sx)+c, m*(mx+2*sx)+c)
                    return xx, m*xx+c, m, c

                fit(uv, h1new-h1fid, ax[1])

            except:
                pass
            if stellar:
                ax[1].text(0.99,-0.003,"z={:.1f}".format(zz), fontdict=font)
            else:
                ax[1].text(0.95,-0.04,"z={:.1f}".format(zz), fontdict=font)

    axar[0, 0].set_xscale('log')
    axar[-1, 0].set_xlabel('k (h/Mpc)', fontdict=font)
    axar[-1, 1].set_xlabel('$\delta_{UV}$', fontdict=font)
    for axis in axar[:, 0]: 
        axis.set_ylabel(r'$b_\Gamma(k)$', fontdict=font)
        axis.set_ylim(-0.5, 0.1)
        axis.set_xlim(0.01, 1)
        axis.legend(prop=fontmanage)
    for axis in axar[:, 1]: 
        axis.set_ylabel(r'$\delta HI_{\Gamma}-\delta HI_{fid}$', fontdict=font)
        if stellar:
            axis.set_ylim(-0.004, 0.004)
            axis.set_xlim(0.985, 1.015)
        else:
            axis.set_ylim(-0.05, 0.05)
            axis.set_xlim(0.91, 1.09)
        axis.legend(loc=1, prop=fontmanage)

    for axis in axar.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            tick.label.set_fontproperties(fontmanage)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    #plt.legend(fontsize=fsize, ncol=1, bbox_to_anchor=(1, 1))

    # and finish up.
    plt.tight_layout()
    if stellar:
        plt.savefig(figpath + 'uvbg_star_n{:02d}.pdf'.format(ncube))
    else:
        plt.savefig(figpath + 'uvbg_n{:02d}.pdf'.format(ncube))
    #




def make_biasauto_plot():
    """"""
    zlist = [3.5, 4.0, 5.0, 6.0][::-1]
    #zlist = [5.0, 6.0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharex='row')
    
    for iz, zz in enumerate(zlist):
        
        print('For redshift = {:.2f}'.format(zz))

        aa = 1.0/(1.0+zz)
        # Put on a fake symbol for the legend.
        
        ph1fid = np.loadtxt(dpath + '/HI_bias_{:.4f}.txt'.format(aa)).T
        plum = np.loadtxt(dpath + 'uvbg/Lum_bias_{:.4f}.txt'.format(aa)).T
        if stellar:
            puv = np.loadtxt(dpath + 'uvbg/UVbg_star_bias_{:.4f}.txt'.format(aa)).T
        else:
            puv = np.loadtxt(dpath + 'uvbg/UVbg_bias_{:.4f}.txt'.format(aa)).T
            
        #
        pkg = puv[1]**2*puv[3]
        pkmg = puv[1]*puv[3]
        pkh1 = ph1fid[2]**2*ph1fid[3]
        b1h1 = ph1fid[1]
        bfid = ph1fid[2][1:6].mean()

        for ip, profile in enumerate([2.0, 2.5, 2.9][::-1]):


            #
            ap = 'ap{:1.0f}p{:1.0f}'.format((profile*10)//10, (profile*10)%10)
            if stellar:
                ph1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_star_bias_{:6.4f}.txt'.format( ap, aa)).T
            else:
                ph1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_bias_{:6.4f}.txt'.format( ap, aa)).T
            k = ph1uv[0]
            pkh1g = ph1uv[2]**2*ph1uv[3]
            bb = np.zeros_like(pkg)
            for i in range(pkg.size):
                bb[i] = solvequad(pkg[i], 2*b1h1[i]*pkmg[i], pkh1[i]-pkh1g[i])
            #bmean = bb[k<0.1][1:].mean()
            bmean = bb[1:6].mean()
            lbl0, lbl1, lbl12 = None, None, None
            if iz == 0:
                lbl0 = r'$\alpha = %.1f$'%profile
                if ip == 0: lbl1, lbl12 = 'Fourier', 'Scatter'
            ax[0].plot(zz, bmean, 'C%do'%ip, label=lbl1)
            bnew = ph1uv[2][1:6].mean()
            ax[1].plot(zz, bnew-bfid, 'C%do'%ip, label=lbl0)

            def fitfunc(p, kmax=6):
                b1, bg = p
                return sum((b1 + bg*puv[1][1:kmax] - ph1uv[1][1:kmax])**2)
            b1new, bg = minimize(fitfunc, [1, 1]).x
            ax[0].plot(zz, bg, 'C%ds'%ip)
            
            ax[1].plot(zz, b1new-bfid, 'C%ds'%ip)
            #ax[1].plot(zz, b1new-bnew, 'C%d<'%ip)



            try:
                if stellar:
                    vals = np.loadtxt(dpath +'/uvbg/scatter_star_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
                else:
                    vals = np.loadtxt(dpath +'/uvbg/scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
                for i in [3, 4, 5, 6, 7]: vals[i] /= vals[i].mean()
                dm, h1fid, h1new, lum, uv = vals[3:]

                def fit(x, y, axis, ic=ip, xlab=None, ylab=None):
                    mx, my, sx, sy = x.mean(), y.mean(), x.std(), y.std()
                    #mask = (x>mx-3*sx) & (x<mx+3*sx)
                    #x, y = x[mask], y[mask]
                    m, c = np.polyfit(x, y, deg=1)
                    linearfit = HuberRegressor().fit(x.reshape(-1, 1), y)
                    m, c = linearfit.coef_[0], linearfit.intercept_
                    return m

                m = fit(uv, h1new-h1fid, ax[1])
                mnew = fit(dm, h1new, ax[1])
                mfid = fit(dm, h1fid, ax[1])
                ax[0].plot(zz, m, 'C%dx'%ip, label=lbl12)
                ax[1].plot(zz, mnew-bfid, 'C%dx'%ip)
            except:
                    pass

    #axar[0, 0].set_xscale('log')
    ax[0].set_ylabel(r'$b_\Gamma$', fontdict=font)
    ax[1].set_ylabel(r'$b_1^\Gamma - b_1$', fontdict=font)
    ax[0].set_ylim(-0.55, 0.15)

    if stellar:
        ax[0].legend(ncol=2, prop=fontmanage, loc=2, frameon=True)
        ax[1].legend(ncol=1, prop=fontmanage, loc=4, frameon=True)
    else:
        ax[0].legend(ncol=2, prop=fontmanage, loc=2, frameon=True)
        ax[1].legend(ncol=1, prop=fontmanage, loc=3, frameon=True)
    for axis in ax[:]: 
        axis.set_xlabel(r'$z$', fontdict=font)
        axis.set_xlim(3.4, 6.4)
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            tick.label.set_fontproperties(fontmanage)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

    #plt.legend(fontsize=fsize, ncol=1, bbox_to_anchor=(1, 1))

    # and finish up.
    plt.tight_layout()
    if stellar:
        plt.savefig(figpath + 'uvbg_bias_star_n{:02d}.pdf'.format(ncube))
    else:
        plt.savefig(figpath + 'uvbg_bias_n{:02d}.pdf'.format(ncube))
    #







def make_bgamma_plot():
    """"""
    zlist = [3.5, 4.0, 5.0, 6.0][::-1]
    #zlist = [5.0, 6.0]

    fig, ax = plt.subplots(1, 2, figsize=(8, 3), sharex='row', sharey='row')
    
    for iz, zz in enumerate(zlist):
        
        print('For redshift = {:.2f}'.format(zz))

        aa = 1.0/(1.0+zz)
        # Put on a fake symbol for the legend.
        
        ph1fid = np.loadtxt(dpath + '/HI_bias_{:.4f}.txt'.format(aa)).T
        bfid = ph1fid[1][1:6].mean()
        k = ph1fid[0]
        pm = ph1fid[3]
        plum = np.loadtxt(dpath + 'uvbg/Lum_bias_{:.4f}.txt'.format(aa)).T
        
        puvstar = np.loadtxt(dpath + 'uvbg/UVbg_star_bias_{:.4f}.txt'.format(aa)).T
        puv = np.loadtxt(dpath + 'uvbg/UVbg_bias_{:.4f}.txt'.format(aa)).T

        #
        
        for ip, profile in enumerate([2.0, 2.5, 2.9][::-1]):

            #
            ap = 'ap{:1.0f}p{:1.0f}'.format((profile*10)//10, (profile*10)%10)
            ph1uvstar = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_star_bias_{:6.4f}.txt'.format( ap, aa)).T
            ph1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_bias_{:6.4f}.txt'.format( ap, aa)).T
            
            lbl0, lbl11, lbl12, lbl13, lbl14 = None, None, None, None, None
            if iz == 0:
                lbl0 = r'$\alpha = %.1f$'%profile
                if ip == 0: lbl11, lbl12, lbl13, lbl14 = 'Cross', 'Auto', 'Scatter', 'Cross'

            kmax = 6

            #Cross bias, both
#            def fitfunc(p, x, y, kmax=10):
#                b1, bg = p
#                wts = k[1:kmax]
#                x, y = x[1:kmax], y[1:kmax]
#                #if abs(b1 - bfid) > bfid: wts = 10000
#                return sum(wts*(b1 + bg*x - y )**2) 
#            tominimize = lambda p: fitfunc(p, puv[1] , ph1uv[1])
#            b1new, bg = minimize(tominimize, [bfid, 1]).x
#            ax[0].plot(zz, bg, 'C%do-'%ip, label=lbl0, alpha=0.7)
#
#            tominimize = lambda p: fitfunc(p, puvstar[1] , ph1uvstar[1])
#            minim = minimize(tominimize, [bfid, 1])
#            if zz == 6.0: print(minim)
#            b1new, bg = minim.x
#            ax[1].plot(zz, bg, 'C%do-'%ip, label=lbl11, alpha=0.7)
#            print(zz, ip, '%.3f'%bg)
#            


            #Cross bias
            def fitfunc(p, x, y, kmax=kmax):
                bg = p
                wts = 1 #k[1:kmax]
                wts = k[1:kmax]
                x, y = x[1:kmax], y[1:kmax]
                #if abs(b1 - bfid) > bfid: wts = 10000
                return sum(wts*(bfid + bg*x - y )**2) 
            tominimize = lambda p: fitfunc(p, puv[1] , ph1uv[1])
            bg = minimize(tominimize, [1]).x
            ax[0].plot(zz, bg, 'C%d*'%ip, label=lbl0)

            tominimize = lambda p: fitfunc(p, puvstar[1] , ph1uvstar[1])
            bg = minimize(tominimize, [1]).x
            ax[1].plot(zz, bg, 'C%d*'%ip, label=lbl14)
            print(zz, ip, '%.3f'%bg)
            


            #Auto bias
            def solveauto(ph1uv, puv, ph1fid):               
                pkuv = puv[1]**2*puv[3]
                pkmuv = puv[1]*puv[3]
                pkh1 = ph1fid[2]**2*ph1fid[3]
                pkh1uv = ph1uv[2]**2*ph1uv[3]
                b1h1 = ph1fid[1]
                bb = np.zeros_like(pkuv)
                for i in range(pkuv.size):
                    bb[i] = solvequad(pkuv[i], 2*b1h1[i]*pkmuv[i], pkh1[i]-pkh1uv[i])
                return bb

            bb = solveauto(ph1uv, puv, ph1fid)
            bmean = bb[1:6].mean()
            ax[0].plot(zz, bmean, 'C%ds'%ip, alpha=0.5)

            bb = solveauto(ph1uvstar, puvstar, ph1fid)
            bmean = bb[1:6].mean()
            ax[1].plot(zz, bmean, 'C%ds'%ip, label=lbl12, alpha=0.5)

            #Scatter bias
            valsstar = np.loadtxt(dpath +'/uvbg/scatter_star_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
            vals = np.loadtxt(dpath +'/uvbg/scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, \
                                                                                                (profile*10)//10, (profile*10)%10, aa)).T
            for i in [3, 4, 5, 6, 7]: vals[i] = vals[i]/vals[i].mean() -1
            for i in [3, 4, 5, 6, 7]: valsstar[i] = valsstar[i]/valsstar[i].mean() -1

            #$fit = HuberRegressor().fit(vals[[3, 7]].T, vals[5])
            #b1new, bg = fit.coef_
            #print('qso, sactter', b1new, bg)
            fit = HuberRegressor().fit(vals[7].reshape(-1, 1), vals[5]-vals[4])
            bg = fit.coef_
            ax[0].plot(zz, bg, 'C%dx'%ip)

            fit = HuberRegressor(epsilon=2.0).fit(valsstar[7].reshape(-1, 1), valsstar[5]-valsstar[4])
            bg = fit.coef_
            ax[1].plot(zz, bg, 'C%dx'%ip, label=lbl13)

    ax[0].text(3.5,-0.49,"QSO", color='k', fontdict=font, alpha=0.8)
    ax[1].text(3.5,-0.49,"QSO+Stellar", color='k', fontdict=font, alpha=0.8)

    #axar[0, 0].set_xscale('log')
    ax[0].set_ylabel(r'$b_\Gamma$', fontdict=font)
    ax[0].set_ylim(-0.55, 0.25)

    ax[0].legend(ncol=2, prop=fontmanage, loc=2, frameon=True)
    ax[1].legend(ncol=2, prop=fontmanage, loc=2, frameon=True)
    for axis in ax[:]: 
        axis.set_xlabel(r'$z$', fontdict=font)
        axis.set_xlim(3.4, 6.4)
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            tick.label.set_fontproperties(fontmanage)
            axis.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))


    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'uvbg_bgamma_n{:02d}.pdf'.format(ncube))
    #







def make_crossratio_plot():
    """"""
    zlist = [3.5, 4.0, 6.0][:]
    #zlist = [5.0, 6.0]

    fig, axar = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey='row')
    
    for iz, zz in enumerate(zlist):
        
        print('For redshift = {:.2f}'.format(zz))
        aa = 1.0/(1.0+zz)
        ax = axar[:, iz]
        #
        ph1fid = np.loadtxt(dpath + '/HI_bias_{:.4f}.txt'.format(aa)).T
        bfid = ph1fid[1][1:6].mean()
        k = ph1fid[0]
        pm = ph1fid[3]
        
        puvstar = np.loadtxt(dpath + 'uvbg/UVbg_star_bias_{:.4f}.txt'.format(aa)).T
        puv = np.loadtxt(dpath + 'uvbg/UVbg_bias_{:.4f}.txt'.format(aa)).T

        mpath = mfpathz(zz)
        
        def bfit(bb, k, buv, b, kmax=10):
            k,  buv = k[1:kmax],  buv[1:kmax]
            #b = b[1:kmax]
            bk = b + bb*np.arctan(k*mpath)/k/mpath
            return sum((bk-buv)**2)

#        def bfit(p, k, buv, kmax=10):
#            b, bb = p
#            k,  buv = k[1:kmax], buv[1:kmax]
#            bk = b + bb*np.arctan(k*mpath)/k/mpath
#            wts = 1
#            return sum((wts*(bk-buv))**2)
##        #
#        
        for ip, profile in enumerate([2.0, 2.5, 2.9][::-1]):

            #
            ap = 'ap{:1.0f}p{:1.0f}'.format((profile*10)//10, (profile*10)%10)
            ph1uvstar = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_star_bias_{:6.4f}.txt'.format( ap, aa)).T
            ph1uv = np.loadtxt(dpath + 'uvbg/HI_UVbg_{}_bias_{:6.4f}.txt'.format( ap, aa)).T
            kmax = 10
            
            lbl = None
            if iz == 2: lbl = r'$\alpha = %.1f$'%profile
            ax[0].plot(k, ph1uv[1]/ph1fid[1], 'C%d'%ip)
            ax[1].plot(k, ph1uvstar[1]/ph1fid[1], 'C%d-'%ip, label=lbl)
            
            ftomin = lambda bb: bfit(bb, k, ph1uv[1], ph1fid[1][1:6].mean(), kmax)
            bb = minimize(ftomin, [0]).x
            ax[0].plot(k, 1+ bb*np.arctan(k*mpath)/k/mpath/ph1fid[1][1:6].mean(), 
                       "C%d--"%ip, lw=2, alpha=0.5, label='$b_I = %.2f$'%bb)

            ftomin = lambda bb: bfit(bb, k, ph1uvstar[1], ph1fid[1][1:6].mean(), kmax)
            bb = minimize(ftomin, [0]).x
            ax[1].plot(k, 1+ bb*np.arctan(k*mpath)/k/mpath/ph1fid[1][1:6].mean(), 
                       "C%d--"%ip, lw=2, alpha=0.5, label='$b_I = %.2f$'%bb)
            print(zz, 'star', bb[0]/bfid)

            for axis in ax: axis.axvline(k[kmax], color='gray', lw=0.2)
            ax[0].text(0.4, 0.9,"$z=%.1f$"%zz, ha='right', color='k', fontdict=font, alpha=0.5)

##
#            ftomin = lambda bb: bfit(bb, k, ph1uv[1],  10)
#            bb = minimize(ftomin, [bfid, 0]).x
#            ax.plot(k, (bb[0]+ bb[1]*np.arctan(k*mpath)/k/mpath)/bfid, "C%d-."%ip, lw=2, alpha=0.5)
#            print(zz, 'qso', bb[0]/bfid)
#
#            ftomin = lambda bb: bfit(bb, k, ph1uvstar[1], 10)
#            bb = minimize(ftomin, [bfid, 0]).x
#            ax.plot(k, (bb[0]+ bb[1]*np.arctan(k*mpath)/k/mpath)/bfid, "C%d:"%ip, lw=2, alpha=0.5)
#            print(zz, 'star', bb[0]/bfid)
#            



    for axis in axar[1]: axis.set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    for axis in axar[:, 0]: axis.set_ylabel(r'$P_{m{\rm HI}^{\Gamma}}(k) / P_{m{\rm HI}^{\rm fid}}(k)$', fontdict=font)
    for axis in axar.flatten(): 
        axis.legend(ncol=1, prop=fontmanage, loc=4, frameon=True)
        axis.set_xlim(0.01, 0.5)
        axis.set_xscale('log')
        #axis.set_yscale('log')
    axar[1, 2].legend(ncol=2, prop=fontmanage,  frameon=True, loc='lower center')
    for axis in axar[0]: axis.set_ylim(0.7, 1.01)
    for axis in axar[1]: axis.set_ylim(0.92, 1.01)

    for axis in axar.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
            tick.label.set_fontproperties(fontmanage)
            #axis.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))


    # and finish up.
    plt.tight_layout()
    plt.savefig(figpath + 'uvbg_crossratio_n{:02d}.pdf'.format(ncube))
    #





if __name__=="__main__":
    #make_biasauto_plot()
    #make_bk_scatter_plot()
    make_bgamma_plot()
    make_crossratio_plot()
    #
