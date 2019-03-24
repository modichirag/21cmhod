import numpy as np
from pmesh.pm import ParticleMesh
from nbodykit.lab import BigFileCatalog, BigFileMesh, FFTPower, ArrayCatalog
from nbodykit.source.mesh.field import FieldMesh
import os, sys
from time import time
from mpi4py import MPI

sys.path.append('./utils')
from uvbg import setupuvmesh


#Get model as parameter
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model name to use', default='ModelA')
parser.add_argument('-s', '--size', help='for small or big box', default='small')
parser.add_argument('-a', '--amp', help='amplitude for up/down sigma 8', default=None)
parser.add_argument('-p', '--profile', help='slope of profile', default=2.9, type=float)
parser.add_argument('-n', '--ncube', type=int, default=2, help='number of voxels')
args = parser.parse_args()

model = args.model 
boxsize = args.size
amp = args.amp
profile = args.profile
ncube = args.ncube


#
#Global, fixed things
scratchyf = '/global/cscratch1/sd/yfeng1/m3127/'
scratchcm = '/global/cscratch1/sd/chmodi/m3127/H1mass/'
project  = '/project/projectdirs/m3127/H1mass/'
cosmodef = {'omegam':0.309167, 'h':0.677, 'omegab':0.048}
alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]
atoz = lambda a: 1/a-1
zzfiles = [round(atoz(aa), 2) for aa in alist]

suff='m1_00p3mh-alpha-0p8-subvol'
ofolder = '../data/outputs/'
outfolder = ofolder + suff


#Parameters, box size, number of mesh cells, simulation, ...
if boxsize == 'small':
    bs, nc, ncsim, sim, prefix = 256, 256, 2560, 'highres/%d-9100-fixed'%2560, 'highres'
elif boxsize == 'big':
    bs, nc, ncsim, sim, prefix = 1024, 1024, 10240, 'highres/%d-9100-fixed'%10240, 'highres'
else:
    print('Box size not understood, should be "big" or "small"')
    sys.exit()

if amp is not None:
    if amp == 'up' or amp == 'dn':
        sim = sim + '-%s'%amp
    else:
        print('Amplitude should be "up" or "dn". Given : ', amp)
        sys.exit()

if bs == 1024: outfolder = outfolder + "-big"
outfolder += "/%s/"%model



if __name__=="__main__":


    pm = ParticleMesh(BoxSize = bs, Nmesh = [nc, nc, nc])
    rank = pm.comm.rank
    wsize = pm.comm.size
    comm = pm.comm
    if rank==0: 
        print(args)
        print(outfolder)


    cube_size = nc/ncube
    shift = cube_size
    cube_length = cube_size*bs/nc

    pmsmall = ParticleMesh(BoxSize = bs/ncube, Nmesh = [cube_size, cube_size, cube_size], dtype=np.float32, comm=MPI.COMM_SELF)
    gridsmall = pmsmall.generate_uniform_particle_grid(shift=0)

    for zz in [6.0]:
        aa = 1/(1+zz)
        cats, meshes = setupuvmesh(zz, suff=suff, sim=sim, pm=pm, profile=profile)
        cencat, satcat = cats
        h1meshfid, h1mesh, lmesh, uvmesh = meshes
        if ncsim == 10240:
            dm = BigFileMesh(scratchyf+sim+'/fastpm_%0.4f/'%aa+\
                            '/1-mesh/N%04d'%nc,'').paint()
        else:  dm = BigFileMesh(project+sim+'/fastpm_%0.4f/'%aa+\
                        '/dmesh_N%04d/1/'%nc,'').paint()
        meshes = [dm, h1meshfid, h1mesh, lmesh, uvmesh]
        names = ['dm', 'h1fid', 'h1new', 'lum', 'uv']
        

        if rank == 0 : print('\nMeshes read\n')
    
        bindexes = np.arange(ncube**3)
        bindexsplit = np.array_split(bindexes, wsize)

        maxload = max(np.array([len(i) for i in bindexsplit]))

        indices = np.zeros((ncube**3, 3))
        totals = np.zeros((ncube**3, len(names)))

        for ibindex in range(maxload):
            if rank == 0: print('For index = %d'%(int(ibindex)))
            try:
                bindex = bindexsplit[rank][ibindex]
                bi, bj, bk = bindex//ncube**2, (bindex%ncube**2)//ncube, (bindex%ncube**2)%ncube
                indices[bindex][0], indices[bindex][1] , indices[bindex][2] = bi, bj, bk 

                bi *= cube_length
                bj *= cube_length
                bk *= cube_length
                print('For rank & index : ', rank, bindex, '-- x, y, z : ', bi, bj, bk)
                poslarge = gridsmall + np.array([bi, bj, bk])
                save = True
            except IndexError:
                poslarge = np.empty((1, 3))
                save = False
                
                
                
            #print(rank, 'Position created')
            layout = pm.decompose(poslarge)
            #if rank == 0 : print(rank, 'Layout decomposed')

            for i in range(len(meshes)):
                vals = meshes[i].readout(poslarge, layout=layout, resampler='nearest').astype(np.float32)
                name = names[i]
                if save:
                    savemesh = pmsmall.paint(gridsmall, mass = vals, resampler='nearest')
                    totals[bindex, i] = savemesh.csum()
                    #print('In rank {:3d}, sum for mesh {:8} is equal to {:.3e}'.format(rank, name, savemesh.csum()))

        indices = comm.gather(indices, root=0)
        totals = comm.gather(totals, root=0)

        if rank ==0:
            indices = np.concatenate([indices[ii][bindexsplit[ii]] for ii in range(wsize)], axis=0)
            totals = np.concatenate([totals[ii][bindexsplit[ii]] for ii in range(wsize)], axis=0)
            tosave = np.concatenate((indices, totals), axis=1)
            #print(tosave)
            header  = 'ix, iy, iz, dm, h1fid, h1new, lum, uv'
            fmt = '%d %d %d %.5e %.5e %.5e %.5e %.5e'
            fname = outfolder + 'scatter_n{:02d}_ap{:1.0f}p{:1.0f}_{:6.4f}.txt'.format(ncube, (profile*10)//10, (profile*10)%10, aa)
            print('Data saved to file {:s}'.format(fname))
            np.savetxt(fname, tosave, header=header, fmt=fmt)
