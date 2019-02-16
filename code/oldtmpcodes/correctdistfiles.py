import numpy as np

alist    = [0.1429,0.1538,0.1667,0.1818,0.2000,0.2222,0.2500,0.2857,0.3333]

    

if __name__=="__main__":
    suff='-m1_00p3mh-alpha-0p8-subvol'
    ofolder = '../data/outputs/'


    for modelname in ['ModelA', 'ModelB']:
        outfolder = ofolder + suff[1:] + "/%s/"%modelname

        for aa in alist:

            tosave = np.loadtxt(outfolder + "HI_dist_{:6.4f}.txt".format(aa))
            print(tosave.shape)
            for i in [0, 2, 3, 4]:
                tosave[:, i]  = tosave[:, i] / (tosave[:, 1] + 1)
            
            header = 'Halo Mass, Number Halos, HI halos, HI centrals, HI satellites'
            np.savetxt(outfolder + "HI_dist_{:6.4f}.txt".format(aa), tosave, fmt='%0.6e', header=header)
