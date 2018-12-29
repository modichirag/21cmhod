import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius


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
    
def sampleilogcdf(n, ilogcdf):
    '''Inverse cdf sampling in log-scale
    '''
    u = np.random.uniform(size=n)
    lu = np.log(u)
    lx = ilogcdf(lu)
    return np.exp(lx)

n = int(1e6)
c = 7
rr = np.array([get_nfw_r(c=c) for i in range(n)])
rric = sampleilogcdf(n, ilogcdfnfw(c=c))
