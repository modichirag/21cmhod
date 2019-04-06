#!/usr/bin/env python3
#
# Set up linear P(k) files.
#
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

db = "../data/"



def Ez2(omm,aa):
    """The evolution parameter (squared) at scale-factor aa,
    normalized to unity at aa=1."""
    w0 = -1.0
    wa = 0.0
    omx= 1.0-omm
    omk= 1.0-omm-omx
    ez = omk/aa/aa+omm/aa/aa/aa
    ez = ez + omx/aa**(3*(1+w0+wa))*np.exp(-3*wa*(1-aa))
    return(ez)
    #




def growthfactor(omm,aa):
    """The growth factor, normalized to 1 today, by which perturbations
    were smaller at aa."""
    w0 = -1.0
    wa = 0.0
    # Set some things.
    NN  = 7500
    lnai= -8.0
    omx = 1.0 - omm
    omk = 1.0 - omm - omx
    # First evolve to aa.  Do integration in log(a)
    phi = 1.0
    phip= 0.0
    hh  = (np.log(aa)-lnai)/NN
    for j in range(NN):
        lna = lnai + (j+0.5)*hh
        aval= np.exp(lna)
        H2  = Ez2(omm,aval)
        omz = omm*np.exp(-3*lna)/H2
        dH2 = omx*np.exp(-3*(1+w0+wa)*lna)
        dH2*= np.exp(-3*wa*(1-aval))*(-3*(1+w0+wa)+3*wa*aval)
        dH2+= omm*np.exp(-3*lna)*(-3)
        dH2+= omk*np.exp(-2*lna)*(-2)
        dp  = phip
        ddp = (1.5*omz-3-0.5*dH2/H2)*phi-(4+0.5*dH2/H2)*phip
        phi +=  dp*hh
        phip+= ddp*hh
    tmp = phi
    # Now evolve all the way to aa=1.0.
    phi = 1.0
    phip= 0.0
    hh  = (np.log(1.0)-lnai)/NN
    for j in range(NN):
        lna = lnai + (j+0.5)*hh
        aval= np.exp(lna)
        H2  = Ez2(omm,aval)
        omz = omm*np.exp(-3*lna)/H2
        dH2 = omx*np.exp(-3*(1+w0+wa)*lna)
        dH2*= np.exp(-3*wa*(1-aval))*(-3*(1+w0+wa)+3*wa*aval)
        dH2+= omm*np.exp(-3*lna)*(-3)
        dH2+= omk*np.exp(-2*lna)*(-2)
        dp  = phip
        ddp = (1.5*omz-3-0.5*dH2/H2)*phi-(4+0.5*dH2/H2)*phip
        phi +=  dp*hh
        phip+= ddp*hh
    # We want the ratio.  Also note that
    # f(om) = ddel/del = (phip+phi)/phi.
    return(tmp/phi*aa)
    #
  




def create_pk(aa,zf=2.0,scale=1.0,\
              pkfile="pk_Planck2018BAO_matterpower_z002.dat"):
    """Generate a P(k) file scaled to scale-factor aa."""
    # First read the P(k) file.
    dd = np.loadtxt(db+pkfile)
    kk = dd[:,0]
    pk = dd[:,1]
    af = 1.0/(1.0 + zf)
    # Scale from redshift zf to the desired scale factor.
    omm= 0.309167
    Df = growthfactor(omm,af)
    Dz = growthfactor(omm,aa)
    pk*= (Dz/Df)**2
    pk*= scale
    ss = Spline(np.log10(kk),np.log10(pk))
    # Now compute and write P(k).
    if np.abs(scale-1.0)<1e-4:
        ff = open("pklin_{:6.4f}.txt".format(aa),"w")
    else:
        ff = open("pklin_{:6.4f}_{:05.3f}.txt".format(aa,scale),"w")
    for kv in np.logspace(np.log10(kk[0]),np.log10(kk[-1]),2048):
        ff.write("{:15.5e} {:15.5e}\n".format(kv,10.**ss(np.log10(kv))))
    ff.close()
    #



if __name__=="__main__":
    #zlist = np.array([2.0,2.5,3.0,4.0,5.0,6.0])
    zlist = np.array([0.0])
    alist = 1.0/(1+zlist)
    for aa in alist:
        create_pk(aa)
        #for scale in [0.95,1.05]:
        #    create_pk(aa,scale=scale)
    #
