#!/usr/bin/env python3
#
# Python script to generate the "index.html" file.
#

import re
import os
import subprocess


# Keywords in the HTML tags.
keyw = ["large-scale structure","intensity mapping","21cm","simulations"]

# Output list.
zlist= [2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
# Items for each redshift.
ilist= ["Halos","Matter","Grids"]


def make_index():
    """Does the work of making the index.html file."""
    # Set up the keyword string.
    keystr= "<META NAME=\"keywords\" CONTENT=\"cosmology"
    for ww in keyw:
        keystr += "," + ww
    keystr+="\">\n"
    #
    with open("index.html","w") as ff:
        ff.write("<HTML>\n")
        ff.write("<HEAD>\n")
        ff.write("<TITLE>Hidden Valley</TITLE>\n")
        ff.write(keystr)
        ff.write("</HEAD>\n")
        ff.write("<BODY BGCOLOR=WHITE>\n\n")
        ff.write("<H2>Hidden Valley simulations</H2>\n\n")
        #
        # We may want to add a figure here?
        #
        ff.write("""
<P>
<DIV ALIGN=CENTER>
<IMG SRC="slices.png" WIDTH=80%><BR>
<I>Slices through one of the Hidden Valley simulations, showing the
projected dark matter (top) and HI (bottom) densities at z=4.
The left panels show a 100Mpc/h thick slice through the full
simulation box, while the middle and right panels show successive zooms
by a factor of 10, centered around the 5th most massive halo.</I><BR>
(From Modi et al. 2019).
</DIV>
<P>
<HR WIDTH=75%>
<P>
        \n""")
        #
        # Put in some text about the simulations.
        #
        ff.write("""
A set of trillion particle simulations in Gpc volumes aimed
at intensity mapping science, described in more detail in
<a href="https://arxiv.org/abs/1904.11923">Modi et al. (2019)</a>.
We make avaialble halo catalogs and subsampled dark matter particles
for each simulation for 9 snapshots between z=2 and 6.<P>

Code to read and process these data are avaialble in
<a href="https://github.com/modichirag/HiddenValleySims">this
GitHub repository</a>.<P>

Please cite <a href="https://arxiv.org/abs/1904.11923">Modi,
Castorina, Feng and White (2019)</a> if you make use of these data.
<P>
<HR WIDTH=75%>
<P>
\n""")
        #
        # Could order these by redshift then item or
        # by item then redshift.  Right now I'm just
        # printing a word, but these will eventually
        # be links to the actual data.
        #
        for isim in ["HV10240-F","HV10240-Fdn","HV10240-Fup","HV10240-R"]:
            ff.write("<h4>"+isim+"</h4>\n\n")
            ff.write("<UL>\n")
            for zz in zlist:
                aa = 1.0/(1.0+zz)
                ff.write("<LI>z={:.1f}\n".format(zz))
                ff.write("  <UL>\n")
                for itm in ilist:
                    if itm=="Halos":
                        link = isim+"_halos_{:06.4f}".format(aa)
                        if os.path.exists(link):
                            ff.write("    <LI><a href=\""+link+"\">"+itm+"</a>\n")
                        else:
                            ff.write("    <LI>"+itm+"\n")
                    else:
                        ff.write("    <LI>"+itm+"\n")
                ff.write("  </UL>\n")
            #
            ff.write("</UL><P>\n\n")
        ff.write("</BODY\n")
        ff.write("</HTML>\n")


if __name__=="__main__":
    make_index()
    subprocess.run(["chmod","ugo+rx","index.html"])
    #
