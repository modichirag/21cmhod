import numpy as np
from scipy.special import spherical_jn
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quadrature

j0 = lambda x: spherical_jn(0, x)


#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='for small or big box', default='big')
args = parser.parse_args()
print(args, args.model)

model = args.model 
boxsize = args.size


alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
suff = 'm1_00p3mh-alpha-0p8-subvol'
if boxsize == 'big':
    suff = suff + '-big'
    bs = 1024
else: bs = 256

outfolder = '../data/outputs/' + suff[:]
outfolder += "/%s/"%model

#


def loginterp(x, y, yint = None, side = "both", lorder = 15, rorder = 15, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6, k=5):
    '''return interpolating function with x, y after extrapolating them with power-law on both sides to absurd range'''   
    if yint is None:
        yint = ius(x, y, k = k)

    if side == "both":
        side = "lr"
        l =lp
        r =rp
    lneff, rneff = 0, 0
    niter = 0
    while (lneff <= 0) or (lneff > 1): #motivated from priors
        lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
        l +=1
        if niter > 100: continue
    print('Left slope = %0.3f at point '%lneff, l)
    niter = 0
    while (rneff < -3) or (rneff > -2):  #motivated from priors
        rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
        r -= 1
        if niter > 100: continue
    print('Rigth slope = %0.3f at point '%rneff, r)

    xl = np.logspace(-18, np.log10(x[l]), 10**6.)
    xr = np.logspace(np.log10(x[r]), 10., 10**6.)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = np.concatenate((xl, xint))
        yint = np.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = np.concatenate((xint, xr))
        yint = np.concatenate((yint, yr))
    yint2 = ius(xint, yint, k = k)

    return yint2


def get_ilpk(aa):
    '''return log-interpolated pk at redshift z(aa) for Hankel transform'''
    pkd = np.loadtxt(outfolder + "HI_bias_{:06.4f}.txt".format(aa))[1:,:]
    kk, pk = pkd[:, 0], pkd[:, 2]**2 * pkd[:, 3]

    #on large scales, extend with b^2*P_lin
    klin, plin = np.loadtxt('../data/pk_Planck2018BAO_matterpower_z000.dat', unpack=True)
    ipklin = ius(klin, plin)

    kt = np.concatenate((klin[(klin < kk[0])], kk))
    pt = np.concatenate((plin[(klin < kk[0])]*pk[0]/ipklin(kk[0]), pk))

    #On small scales, truncate at k=1
    pt = pt[kt<1]
    kt = kt[kt<1]
    ilpk = loginterp(kt, pt)
    return ilpk 


def xij0f(k, r, ipk):
    '''return integrand for hankel transform with j0'''
    return k**3*ipk(k)/2/np.pi**2 *j0(k*r)/k
    

if __name__=="__main__":

    r = np.linspace(1, 120, 5)
    for iz, aa in enumerate(alist):
        zz = 1/aa-1
        print('For redshift : z = ', zz)

        xi = np.zeros_like(r)
        ilpk = get_ilpk(aa)

        for i in range(r.size):
            f = lambda k: xij0f(k, r[i], ilpk)
            xi[i] = quadrature(f, 1e-6, 1e2, maxiter=5000)[0]
            if i%10 == 0: print(i)

        np.savetxt(outfolder + "HI_hankelxi_{:06.4f}.txt".format(aa), np.stack((r, xi)).T, header='r, xi')
