#!/usr/bin/env python3
#
# Python script to generate symbolic links to the /glagol
# file system for the relevant files.
#

import re
import os
import glob
import subprocess


glagol = "/glagol/HiddenValley/"


# Output list.
zlist= [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
zlist= [2.0]


def make_link(isim):
    """Makes a symbolic link from /glagol to the local directory."""
    for zz in zlist:
        aa  = 1.0/(1.0+zz)
        sd = glagol + isim + "/fastpm_{:06.4f}".format(aa)
        # First do the halos.
        hh  = sd + "/LL-0.200"
        td  = isim+"_halos_{:06.4f}".format(aa)
        if not os.path.exists(td):
            print("Making: ",td)
            os.mkdir(td)
        for fn in ["header","attr-v2",\
                   "Position","Velocity","ID","InitialPosition","Vdisp"]:
            src = hh+"/"+fn
            tar = td+"/"+fn
            print(src,tar)
            if os.path.exists(src) and not os.path.exists(tar):
                print("Linking: ",tar)
                subprocess.run(["ln","-s",src,tar])
    #


if __name__=="__main__":
    for isim in ["HV10240-F","HV10240-Fdn","HV10240-Fup","HV10240-R"]:
        make_link(isim)
    #
