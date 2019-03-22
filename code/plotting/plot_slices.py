#!/usr/bin/env python3
#
# Makes a figure showing slices through the DM and HI
# fields in the simulation.
#
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import glob



# I am thinking of centering the slice(s) on the 5th largest halo
# in the simulation at z=4, then plotting arcsinh-scaled densities
# over the full Gpc, then 100Mpc then 10Mpc (i.e. zooming by 10x
# going from left to right).  Top row DM, bottom row HI.

# Paths to the matter and halo data on NERSC:
dbm="/global/cscratch1/sd/yfeng1/m3127/highres/10240-9100-fixed/"+\
    "visualslice_0.2000/1S-HID-0005/Position/"
dbh="/global/cscratch1/sd/yfeng1/m3127/highres/10240-9100-fixed/"+\
    "visualslice_0.2000/LL-0.200S-HID-0005/Position/"






def HI_mass(mhalo,aa):
    """Makes a 21cm "mass" from a box of halo masses."""
    zp1 = 1.0/aa
    zz  = zp1-1
    # Set the parameters of the HOD, using the "simple" form.
    #   MHI ~ M0  x^alpha Exp[-1/x]       x=Mh/Mmin
    # from the Appendix of https://arxiv.org/pdf/1804.09180.pdf, Table 6.
    # Fits valid for 1<z<6:
    mcut= 1e10*(6.11-1.99*zp1+0.165*zp1**2)
    alp = (1+2*zz)/(2+2*zz)
    # Work out the HI mass/weight per halo -- ignore prefactor.
    xx  = mhalo/mcut+1e-10
    mHI = xx**alp * np.exp(-1/xx)
    # Scale to some standard number in the right ball-park.
    mHI*= 2e9*np.exp(-1.9*zp1+0.07*zp1**2)
    # Return the HI masses.
    return(mHI)
    #







def make_slice_plot(aa=0.2000,Npix=128,Lbox=1024.):
    """Does the work of making the slice figure."""
    # Set up some parameters.
    Npart  = 10240.
    mpart  = 0.3*2.7755e11*(Lbox/Npart)**3
    scales = [1.0,10.0,100.0]		# The zoom levels.
    dscale = [1.0, 5.,25.]		# The density scaling for the arcsinh.
    dmaxs  = [5.0,10.,100]		# The density scaling for the arcsinh.
    # Work out the centering position.
    ff=open(dbh+"000000","rb")
    pos=np.fromfile(ff,dtype=('f4',3))
    ff.close()
    pos/=Lbox
    ctr=pos[0,:]
    print("Centering at ",ctr)
    # Set the bounds.
    xmin,xmax = -0.5000,0.5000
    ymin,ymax = -0.5000,0.5000
    zmin,zmax = -5/Lbox,5/Lbox
    area      = (xmax-xmin)*(ymax-ymin)
    # Now make the figure.
    clab   = 'silver'	# Color for the label.
    cbar   = 'silver'	# Color for the scale bar.
    fig,ax = plt.subplots(2,3,figsize=(6,4),sharex=True,sharey=True)
    for ix in range(ax.shape[0]):
        for iy in range(ax.shape[1]):
            ii = ix*ax.shape[1] + iy
            ss = scales[ii%len(scales)]
            dd = np.zeros( (Npix,Npix) ,dtype='float')
            # Read the data from file.  We re-read the data each time,
            # which is I/O inefficient but reduces the memory footprint.
            if ix==0:	# DM.
                ntotal=0
                for fn in sorted(glob.glob(dbm+"0000*")):
                    ff=open(fn,"rb")
                    pos=np.fromfile(ff,dtype=('f4',3))
                    ff.close()
                    pos/=Lbox
                    # Periodically rewrap around ctr.
                    rr = pos-ctr
                    rr[rr<-0.5] += 1.0
                    rr[rr> 0.5] -= 1.0
                    # Trim to the slice in z -- this is always the same
                    # width.  Also keep track of how many particles pass.
                    ww=np.nonzero( (rr[:,2]>zmin)&(rr[:,2]<zmax) )[0]
                    rr=rr[ww,:2]
                    ntotal+=rr.shape[0]
                    # Scale the data and cut to the required region (area).
                    rr *= ss	# Scale.
                    yy  = np.histogram2d(rr[:,0]+0.5,rr[:,1]+0.5,\
                                         bins=Npix,range=[[0,1],[0,1]])[0]
                    dd += yy.astype('float')
                # Assume the slice as a whole is at mean density.
                mdens = ntotal*area/ss**2/float(Npix**2)
                dd    = dd/mdens
                print(ii,ntotal,mdens,np.min(dd),np.max(dd))
            else:	# HI
                ntotal=0
                for fn in sorted(glob.glob(dbh+"0000*")):
                    ff=open(fn,"rb")
                    pos=np.fromfile(ff,dtype=('f4',3))
                    ff.close()
                    pos/=Lbox
                    ff=open(fn.replace("Position","Length"),"rb")
                    mh=np.fromfile(ff,dtype='i4') * mpart
                    ff.close()
                    # Periodically rewrap around ctr.
                    rr = pos-ctr
                    rr[rr<-0.5] += 1.0
                    rr[rr> 0.5] -= 1.0
                    # Trim to the slice in z -- this is always the same
                    # width.  Also keep track of how many halos pass.
                    ww=np.nonzero( (rr[:,2]>zmin)&(rr[:,2]<zmax) )[0]
                    rr=rr[ww,:2]
                    mh=mh[ww]
                    # Scale the data and cut to the required region (area).
                    mHI = HI_mass(mh,aa)
                    rr *= ss	# Scale.
                    yy  = np.histogram2d(rr[:,0]+0.5,rr[:,1]+0.5,\
                                         weights=mHI,\
                                         bins=Npix,range=[[0,1],[0,1]])[0]
                    dd += yy.astype('float')
                    ntotal+= np.sum(mHI)
                # Assume the slice as a whole is at mean density.
                mdens = ntotal*area/ss**2/float(Npix**2)
                dd    = dd/mdens
                print(ii,ntotal,mdens,np.min(dd),np.max(dd))
            # Plot the data, scaled.
            #ax[ix,iy].imshow(np.arcsinh(dd.T/dscale[ii%len(dscale)]),\
            #                 aspect='equal',extent=[0,1,0,1],\
            #                 vmin=0,vmax=np.arcsinh(5.0))
            #ax[ix,iy].imshow(np.arcsinh(dd.T),\
            #                 aspect='equal',extent=[0,1,0,1],\
            #                 vmin=0,vmax=np.arcsinh(dmaxs[ii%len(dmaxs)]), cmap='hot')
            ax[ix,iy].imshow(np.log(1+dd.T),\
                             aspect='equal',extent=[0,1,0,1],\
                             vmin=0,vmax=np.log(1+dd.max()), cmap='hot')
            # Put on a scale bar.
            xmin,xmax,yval = 0.10,0.10+0.1*(1e3/Lbox),0.95
            scale_text     = r'$\mathbf{'+"{:.0f}".format(100./ss)+\
                             r'\,h^{-1}}$Mpc'
            ax[ix,iy].plot([xmin,xmax],[yval,yval],'-',color=cbar,lw=2)
            ax[ix,iy].text(xmin,yval-0.05,scale_text,fontweight='bold',\
                           ha='left',va='top',color=cbar)
            # Tidy up the plot.
            ax[ix,iy].set_aspect('equal')
            ax[ix,iy].set_xticks([])
            ax[ix,iy].set_yticklabels('')
            ax[ix,iy].set_xticks([])
            ax[ix,iy].set_yticklabels('')
            #
    # Put on some more labels.
    ax[0,0].text(0.05,0.05,r'DM',color=clab,fontweight='bold')
    ax[1,0].text(0.05,0.05,r'HI',color=clab,fontweight='bold')
    # and finish up.
    plt.tight_layout()
    plt.savefig('../../figs/slices.pdf')
    #






if __name__=="__main__":
    make_slice_plot()
    #
