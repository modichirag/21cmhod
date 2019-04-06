#!/usr/bin/env python3
#
# Plots the power spectra and Fourier-space biases for the HI.
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline as Spline
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.signal import savgol_filter
from mcfit import P2xi
import sys
sys.path.append('../utils/')
from tools import loginterp

##
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

##
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='which box size simulation', default='big')
args = parser.parse_args()
if args.model == None:
    import sys
    print('Specify a model name')
    sys.exit()
print(args, args.model)

##########################################
model = args.model #'ModelD'
boxsize = args.size

suff = 'm1_00p3mh-alpha-0p8-subvol'
bs = 256
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
dpath = '../../data/outputs/%s/%s/'%(suff, model)
figpath = '../../figs/%s/'%(suff)
try: os.makedirs(figpath)
except: pass


svfilter = True
if bs == 256: winsize = 7
elif bs == 1024: winsize  = 19
polyorder = 3


zlist = [2.0, 4.0, 6.0]
#b1lst = [0.900,2.750]
#b2lst = [0.800,6.250]
#alpha = [1.500,0.145]
b1lst = [0.900, 1.58, 2.70]
b2lst = [0.800, 2.0, 6.250]
alpha = [1.500, 0.7, 0.145]
alphad = dict(('%0.1f'%zlist[i], alpha[i]) for i in range(len(zlist)))
b1lstd = dict(('%0.1f'%zlist[i], b1lst[i]) for i in range(len(zlist)))
b2lstd = dict(('%0.1f'%zlist[i], b2lst[i]) for i in range(len(zlist)))

def get_ilpk(zz, mode='real'):
    '''return log-interpolated pk at redshift z(aa) for Hankel transform'''
    aa = 1/(zz+1)
    al, b1, b2 = alphad['%0.1f'%zz], b1lstd['%0.1f'%zz], b2lstd['%0.1f'%zz], 

    pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
    bb = pkd[1:6, 2].mean()
    print(1+b1, bb, (1+b1)/ bb)
    if mode == 'real':
        kk, pk = pkd[:, 0], pkd[:, 2]**2 * pkd[:, 3]
    elif mode == 'red':
        pks = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
        kk, pk = pks[:, 0], pks[:, 1]
        bbs = (pk/ius(pkd[:, 0], pkd[:, 3])(kk))**0.5
        bb = bbs[1:6].mean()
        #pk *= bb**2/bbs**2

    #on large scales, extend with b^2*P_lin
    klin, plin = np.loadtxt('../../data/pklin_{:06.4f}.txt'.format(aa), unpack=True)
    plin *= bb**2
    ipklin = ius(klin, plin)
    kt = np.concatenate((klin[(klin < kk[0])], kk))
    pt = np.concatenate((plin[(klin < kk[0])]*pk[0]/ipklin(kk[0]), pk))

    #On small scales, truncate at k=1
    pt = pt[kt<1]
    kt = kt[kt<1]

    ilpk = loginterp(kt, pt)
    return ilpk
#    #Get ZA
#    pkz = np.loadtxt("../../theory/zeld_{:6.4f}.pkr".format(aa))
#    kk  = pkz[:,0]
#    pzh = (1+al*kk**2)*pkz[:,1]+b1*pkz[:,2]+b2*pkz[:,3]+\
#          b1**2*pkz[:,4]+b2**2*pkz[:,5]+b1*b2*pkz[:,6]
#    kz = np.concatenate((klin[(klin < kk[0])], kk))
#    pz = np.concatenate((plin[(klin < kk[0])]*pzh[0]/ipklin(kk[0]), pzh))
#
#    ilpk = loginterp(kt, pt)
#    ilpklin = loginterp(klin, plin)
#    ilpkza = loginterp(kz, pz)
#    return ilpk, ilpklin, ilpkza 
#


def make_bao_plot(fname):
    """Does the work of making the BAO figure."""

    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(8,4))

    jj = 0
    for iz, zz in enumerate(zlist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        a0, b1, b2 = alphad['%0.1f'%zz], b1lstd['%0.1f'%zz], b2lstd['%0.1f'%zz], 

        #PS
        pkd = np.loadtxt(dpath + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
        bb = pkd[1:6, 2].mean()
        #redshift space
        pks = np.loadtxt(dpath + "HI_pks_1d_{:06.4f}.txt".format(aa))[1:,:]
        pks = ius(pks[:,0],pks[:,1])(pkd[:,0])
        
        # Now read linear theory and put it on the same grid -- currently
        # not accounting for finite bin width.
        lin = np.loadtxt("../../data/pklin_{:6.4f}.txt".format(aa))
        lin = ius(lin[:,0],lin[:,1])(pkd[:,0])
        
        ps = pkd[:,2]**2 *pkd[:,3]
        ss  = savgol_filter(ps, winsize,polyorder=2)
        rat = ps/ss 
        ss  = savgol_filter(pks, winsize ,polyorder=2)
        rats = pks/ss 
        ss  = savgol_filter(lin,winsize,polyorder=2)
        ratlin = lin/ss 

        ax[0].plot(pkd[:,0],rat+0.2*(jj),'C%d-'%iz,\
                    label="z={:.1f}".format(zz))
        ax[0].plot(pkd[:,0],rats+0.2*(jj),'C%d--'%iz, alpha=0.5, lw=2)
        ax[0].plot(pkd[:,0],ratlin+0.2*(jj),':', color='gray',  lw=1.5)
        ax[0].axhline(1+0.2*(jj),color='gray', lw=0.5, alpha=0.5)

        #xi
        
        #ilpk, ilpklin, ilpkza = get_ilpk(zz)
        ilpk = get_ilpk(zz)
        ilpks = get_ilpk(zz, mode='red')

        kk = np.logspace(-4, 2, 10000)
        xif = P2xi(kk)
        rr, xi = xif(ilpk(kk))
        rr, xis = xif(ilpks(kk))
        mask = (rr > 1) & (rr < 150)
        rr, xi, xis = rr[mask], xi[mask], xis[mask]
        #read theory
        xiz = np.loadtxt("../../theory/zeld_{:6.4f}.xir".format(aa)).T
        rrz = xiz[0]
        #xilin = xiz[1]*(1+b1)**2
        xilin = xiz[1]*(bb)**2
        ff = (0.31/(0.31+0.69*aa**3))**0.55
        ffb = ff/(1+b1)
        #kaiser = (1+b1)**2*(1 + 2*ffb/3 + ffb**2/5)
        kaiser = (bb)**2*(1 + 2*ffb/3 + ffb**2/5)
        xilins = xiz[1]*kaiser
        #interpolate theory on data
        xilinrr = ius(rrz, xilin)(rr)
        xilinsrr = ius(rrz, xilins)(rr)

        off = 10
        ax[1].plot(rr, jj*off + rr**2*xi, 'C%d'%iz, label="z={:.1f}".format(zz))
        ax[1].plot(rr, jj*off + rr**2*xis, 'C%d--'%iz, lw=2, alpha=0.5)
        ax[1].plot(rrz, jj*off + xilin, ':', color='gray', lw=1.5)
        ax[1].plot(rrz, jj*off + xilins, '-', color='gray', lw=0.1)

        jj =  jj+1

#        ax[1].plot(rr, 0.2*jj + rr**2*xi/xilinrr, 'C%d'%iz, label="z={:.1f}".format(zz))
#        ax[1].plot(rr, 0.2*jj + rr**2*xis/xilinsrr, 'C%d--'%iz, lw=2, alpha=0.7)

        #ZA
#        xiza = xiz[2] + b1*xiz[3] + b2*xiz[4] + b1**2*xiz[5]+\
#                   b1*b2*xiz[6] + b2**2*xiz[7] + a0*xiz[8]
#        xizas = xiz[2+9] + b1*xiz[3+9] + b2*xiz[4+9] + b1**2*xiz[5+9]+\
#                   b1*b2*xiz[6+9] + b2**2*xiz[7+9] + a0*xiz[8+9]
#        ax[1].plot(rrz, jj*10+ xiza, 'C%d--'%iz, lw=2, alpha=0.5)
#        ax[1].plot(rrz, jj*10+ xizas, 'k:', lw=2, alpha=0.5)
        


    # Tidy up the plot.
    ax[0].legend(ncol=2,framealpha=0.5,prop=fontmanage)
    ax[0].set_xlim(0.045,0.4)
    ax[0].set_ylim(0.75,1.5)
    #ax[1].legend(ncol=2,framealpha=0.5,prop=fontmanage)
    ax[1].set_xlim(55, 120)
    ax[1].set_ylim(0., 37)
    #ax[1].set_ylim(0.75, 5)
    for ii in range(ax.size):
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')
    # Put on some more labels.
    ax[0].set_xlabel(r'$k\quad [h\,{\rm Mpc}^{-1}]$', fontdict=font)
    ax[1].set_xlabel(r'$r\quad [{\rm Mpc}/h]$', fontdict=font)
    ax[0].set_ylabel(r'$P(k)/P_{\rm nw}(k)$+offset', fontdict=font)
    ax[1].set_ylabel(r'$r^2 \xi(r)$+offset', fontdict=font)
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)


    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #






def make_bao_xi_plot(fname):
    """Does the work of making the BAO figure."""
    clist = ['b','c','g','m','r','y']
    # Now make the figure.
    fig,ax = plt.subplots(1,2,figsize=(8,4),sharey=True)
    ii,jj  = 0,0

    for iz, zz in enumerate(zlist):
        # Read the data from file.
        aa  = 1.0/(1.0+zz)
        a0, b1, b2 = alphad['%0.1f'%zz], b1lstd['%0.1f'%zz], b2lstd['%0.1f'%zz], 
        
        #ilpk, ilpklin, ilpkza = get_ilpk(zz)
        ilpk = get_ilpk(zz)
        ilpks = get_ilpk(zz, mode='red')
        kk = np.logspace(-4, 2, 10000)
        xif = P2xi(kk)
        rr, xi = xif(ilpk(kk))
        rr, xis = xif(ilpks(kk))
        mask = (rr > 1) & (rr < 150)
        rr, xi, xis = rr[mask], xi[mask], xis[mask]
        #read theory
        xiz = np.loadtxt("../../theory/zeld_{:6.4f}.xir".format(aa)).T
        rrz = xiz[0]
        xilin = xiz[1]*(1+b1)**2
        ff = (0.31/(0.31+0.69*aa**3))**0.55
        ffb = ff/(1+b1)
        kaiser = (1+b1)**2*(1 + 2*ffb/3 + ffb**2/5)
        xilins = xiz[1]*kaiser
        #interpolate theory on data
        xilinrr = ius(rrz, xilin)(rr)
        xilinsrr = ius(rrz, xilins)(rr)

        xiza = xiz[2] + b1*xiz[3] + b2*xiz[4] + b1**2*xiz[5]+\
                   b1*b2*xiz[6] + b2**2*xiz[7] + a0*xiz[8]
        xizas = xiz[2+9] + b1*xiz[3+9] + b2*xiz[4+9] + b1**2*xiz[5+9]+\
                   b1*b2*xiz[6+9] + b2**2*xiz[7+9] + a0*xiz[8+9]
        ax[ii].plot(rr, rr**2*xi, 'C%d'%iz, label="z={:.1f}".format(zz))
        if iz == 0:
            lbls = ['redshift sim', 'real lin', 'real za', 'redshift za']
        else: lbls = [None, None, None, None]
        ax[ii].plot(rr, rr**2*xis/kaiser*(1+b1)**2, 'C%d--'%iz, lw=2, alpha=0.5, label=lbls[0])
        ax[ii].plot(rrz, xilin, 'k:', label=lbls[1])
        ax[ii].plot(rrz, xiza, 'gray', ls="--", lw=2, alpha=0.5, label=lbls[2])
        ax[ii].plot(rrz, xizas/kaiser*(1+b1)**2, 'gray', ls=":", lw=2, alpha=0.5, label=lbls[3])
        #ax[ii].axhline(1+0.2*(jj//2),color='gray', lw=0.5, alpha=0.5)

        ii = (ii+1)%2
        jj =  jj+1
    # Tidy up the plot.
    for ii in range(ax.size):
        ax[ii].legend(ncol=2,framealpha=0.5,prop=fontmanage)
        ax[ii].set_xlim(1, 120)
        #ax[ii].set_ylim(1, 30)
        ax[ii].set_xscale('linear')
        ax[ii].set_yscale('linear')

    # Put on some more labels.
    for axis in ax: axis.set_xlabel(r'$r\quad [{\rm Mpc}/h]$', fontdict=font)
    ax[0].set_ylabel(r'$r^2 \xi(r)$', fontdict=font)
    for axis in ax.flatten():
        for tick in axis.xaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)
        for tick in axis.yaxis.get_major_ticks():
            tick.label.set_fontproperties(fontmanage)


    # and finish up.
    plt.tight_layout()
    plt.savefig(fname)
    #


if __name__=="__main__":
    make_bao_plot(figpath + 'HI_bao_%s.pdf'%model)
    make_bao_xi_plot(figpath + 'HI_bao_xi_%s.pdf'%model)
    #
