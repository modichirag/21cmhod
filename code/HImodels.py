import numpy as np
from nbodykit.utils import DistributedArray
from nbodykit.lab import BigFileCatalog, MultipleSpeciesCatalog
import hodanalytics as hodanals

class ModelA():
    
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.normhalo = 3e5*(1+(3.5/self.zz)**6)
        self.normsat = self.normhalo*(1.75 + 0.25*self.zz)


    def assignHI(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = self.assignsat(satcat['Mass'].compute())
        mHIcen = self.assigncen(mHIhalo, mHIsat, satcat['GlobalID'].compute(), 
                                cencat.csize, cencat.comm)
        
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        xx  = msat/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normsat
        return mHI
        
    def assigncen(self, mHIhalo, mHIsat, satid, censize, comm):
        #Assumes every halo has a central...which it does...usually
        da = DistributedArray(satid, comm)
        mHI = da.bincount(mHIsat, shared_edges=False)
        zerosize = censize - mHI.cshape[0]
        zeros = DistributedArray.cempty(cshape=(zerosize, ), dtype=mHI.local.dtype, comm=comm)
        zeros.local[...] = 0
        mHItotal = DistributedArray.concat(mHI, zeros, localsize=mHIhalo.size)
        return mHIhalo - mHItotal.local
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='galaxies', position='RSDpos', weight='HImass'):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh
        
    


class ModelA2(ModelA):
    '''Same as model A with a different RSD for satellites
    '''
    def __init__(self, aa):

        super().__init__(aa)
        
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity_HI']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos

        

        
    
class ModelB():
    
    def __init__(self, aa, hodparams=None):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.normhalo = 3e5*(1+(3.5/self.zz)**6)
        self.normsat = self.normhalo*(1.75 + 0.25*self.zz)
        self.hodparams = hodparams
        
    def assignhalo(self, mhalo):
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        xx  = msat/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normsat
        return mHI
        
    def assigncen(self, mhalo, hodparams = None, mh1halo=None):
        #Assumes every halo has a central...which it does...usually
        if hodparams is None: hodparams = self.hodparams
        if mh1halo is None: mh1halo = self.assignhalo(mhalo)
        h1hsatparams = [self.mcut, self.alp, self.normsat]
        h1msat = hodanals.mtotsatHImodelA(mhalo, hodparams, h1hsatparams)
        return mh1halo - h1msat
        


    def moster(Minf,z):
        """
        moster(Minf,z):
        Returns the stellar mass (M*) given Minf and z from Table 1 and
        Eq. (2,11-14) of Moster++13 [1205.5807].
        This version works in terms of Msun units, not Msun/h.
        To get "true" stellar mass, add 0.15 dex of lognormal scatter.
        To get "observed" stellar mass, add between 0.1-0.45 dex extra scatter.
        """
        zzp1  = z/(1+z)
        M1    = 10.0**(11.590+1.195*zzp1)
        mM    = 0.0351 - 0.0247*zzp1
        beta  = 1.376  - 0.826*zzp1
        gamma = 0.608  + 0.329*zzp1
        Mstar = 2*mM/( (Minf/M1)**(-beta) + (Minf/M1)**gamma )
        Mstar*= Minf
        return(Mstar)
        #





class ModelC():
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity but do not have any dispersion over it
    '''
    def __init__(self, aa):

        self.aa = aa
        self.zz = 1/aa-1

        self.alp = (1+2*self.zz)/(2+2*self.zz)
        self.mcut = 1e9*( 1.8 + 15*(3*self.aa)**8 )
        self.normhalo = 3e5*(1+(3.5/self.zz)**6)
        self.normsat = 0


    def assignHI(self, halocat, cencat, satcat):
        mHIhalo = self.assignhalo(halocat['Mass'].compute())
        mHIsat = np.zeros(satcat['Mass'].size)
        mHIcen = np.zeros(cencat['Mass'].size) 
        
        return mHIhalo, mHIcen, mHIsat
        
        
    def assignhalo(self, mhalo):
        xx  = mhalo/self.mcut+1e-10
        mHI = xx**self.alp * np.exp(-1/xx)
        mHI*= self.normhalo
        return mHI

    def assignsat(self, msat):
        return msat*0
        
    def assigncen(self, mcen):
        return mcen*0
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        hrsdpos = halocat['Position']+halocat['Velocity']*los * rsdfac
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


    def createmesh(self, bs, nc, halocat, cencat, satcat, mode='halos', position='RSDpos', weight='HImass'):
        '''use this to create mesh of HI
        '''
        comm = halocat.comm
        if mode == 'halos': catalogs = [halocat]
        elif mode == 'galaxies': catalogs = [cencat, satcat]
        elif mode == 'all': catalogs = [halocat, cencat, satcat]
        else: print('Mode not recognized')
        
        rankweight       = sum([cat[weight].sum().compute() for cat in catalogs])
        totweight        = comm.allreduce(rankweight)

        for cat in catalogs: cat[weight] /= totweight/float(nc)**3            
        allcat = MultipleSpeciesCatalog(['%d'%i for i in range(len(catalogs))], *catalogs)
        mesh = allcat.to_mesh(BoxSize=bs,Nmesh=[nc,nc,nc],\
                                 position=position,weight=weight)

        return mesh




class ModelD(ModelC):
    '''Vanilla model with no centrals and satellites, only halo
    Halos have the COM velocity and a dispersion from VN18 added over it
    '''
    def __init__(self, aa):

        super().__init__(aa)
        self.vdisp = self._setupvdisp()


    def _setupvdisp(self):
        vzdisp0 = np.array([31, 34, 39, 44, 51, 54])
        vzdispal = np.array([0.35, 0.37, 0.38, 0.39, 0.39, 0.40])
        vdispz = np.arange(0, 6)
        vdisp0fit = np.polyfit(vdispz, vzdisp0, 1)
        vdispalfit = np.polyfit(vdispz, vzdispal, 1)
        vdisp0 = self.zz * vdisp0fit[0] + vdisp0fit[1]
        vdispal = self.zz * vdispalfit[0] + vdispalfit[1]
        return lambda M: vdisp0*(M/1e10)**vdispal
        
      
    def assignrsd(self, rsdfac, halocat, cencat, satcat, los=[0,0,1]):
        dispersion = np.random.normal(0, self.vdisp(halocat['Mass'].compute())).reshape(-1, 1)
        hvel = halocat['Velocity']*los + dispersion*los
        hrsdpos = halocat['Position']+ hvel*rsdfac
        
        crsdpos = cencat['Position']+cencat['Velocity']*los * rsdfac
        srsdpos = satcat['Position']+satcat['Velocity']*los * rsdfac
        return hrsdpos, crsdpos, srsdpos


