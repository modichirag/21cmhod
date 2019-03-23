import numpy as np
from scipy.special import spherical_jn
from scipy.misc import derivative
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quadrature

import sys
sys.path.append('../utils/')
from tools import loginterp

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
            if i%1 == 0: print(i, r[i])
            f = lambda k: xij0f(k, r[i], ilpk)
            xi[i] = quadrature(f, 1e-5, 1e2, maxiter=1000)[0]

        np.savetxt(outfolder + "HI_hankelxi_{:06.4f}.txt".format(aa), np.stack((r, xi)).T, header='r, xi')
