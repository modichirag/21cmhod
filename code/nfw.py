B1;5202;0cimport numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import interp1d, interp2d
from time import time

#  float get_nfw_r(const float c) const {
#     // Returns r/rvir, using a simple sampling strategy
#     float   x;
#     while (1==1) {
#       x = drand48() * c;
#       if (drand48() < 4*x/(1+x)/(1+x)) { /* Compare to r^2.rho(r) */
#         return(x/c);
#       }
#     }
#   }

def get_nfw_r(c=7):
    '''Martin's code based on rejection sampling
    '''
    x = np.random.uniform() * c
    while (np.random.uniform() > 4*x/(1+x)/(1+x)):
        x = np.random.uniform() * c
    else:
        return(x/c)

###############################################

def gcum(x):
    return np.log(1+x) - x/(1+x)
    
def cumnfw(r, c=7):
    '''cumulative pdf for nfw profile at scaled radius(by r_vir)=r
    Taken from https://halotools.readthedocs.io/en/latest/source_notes/empirical_models/phase_space_models/nfw_profile_source_notes.html
    '''
    return gcum(r*c)/gcum(c)

def ilogcdfnfw(c=7):
    '''Inverse cdf of nfw in log-scale
    '''
    rr = np.logspace(-3, 0, 1000)
    cdf = cumnfw(rr, c=c)
    lrr, lcdf = np.log(rr), np.log(cdf)
    return ius(lcdf, lrr)
    
def ilog2dcdfnfw():
    '''Create 2D interpolating function between conc, cdf => r
    '''
    civ = np.linspace(5, 10, 100)
    cdfiv = np.logspace(-5, 0, 1000)
    riv = []
    for i , c in enumerate(civ):
        riv.append(ilogcdfnfw(c)(np.log(cdfiv)))
    riv = np.array(riv)
    return interp2d(civ, np.log(cdfiv), riv.T)    


def sampleilogcdf(n, ilogcdf):
    '''Inverse cdf sampling in log-scale
    '''
    u = np.random.uniform(size=n)
    lu = np.log(u)
    lx = ilogcdf(lu)
    return np.exp(lx)

####


if __name__=="__main__":

    nhalo = int(1e5)
    ratio_sat_cen=100
    cc = np.random.uniform(low=5, high=10, size=nhalo)
    nn =[]
    for i, c in enumerate(cc):
        nn.append(np.random.randint(1, 2*ratio_sat_cen))
    print('Total number of halos = ', nhalo)
    print('Total number of satellites = ', np.array(nn).sum())
    print('Ratio of satellites to central = ', np.array(nn).sum()/nhalo)

    #
    start = time()
    for i, c in enumerate(cc):
        n = nn[i]
        rr = [get_nfw_r(c) for i in range(n)]    
    end = time()
    print('Rejection sampling time', end - start)
    #
    start = time()
    for i, c in enumerate(cc):
        n = nn[i]
        ilcdf =  ilogcdfnfw(c)
        rr = sampleilogcdf(n, ilcdf)
    end = time()
    print('Inverse sampling with creating icdf(c)', end - start)
    #
    ilcdf2d = ilog2dcdfnfw()
    start = time()
    for i, c in enumerate(cc):
        n = nn[i]
        ilcdf = lambda x: ilcdf2d(c, x)
        rr = sampleilogcdf(n, ilcdf)
    end = time()
    print('Inverse sampling time with 2D spline', end - start)
