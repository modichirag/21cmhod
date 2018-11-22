#!/bin/bash

aa=(0.0100 0.0909 0.1111 0.1667 0.2000 0.2500 0.2857)
bs=2048
#aa=(0.0100 0.0909 0.1111 0.1667 0.2000 0.2500)



#aa=(0.0100 0.0900 )
for i in "${aa[@]}" ;
do
    echo Doing for $i
    python halobias.py ./$bs-test-0.168/power_$i.txt --verbose --nmesh=256 --nn=10 --with-plot /global/cscratch1/sd/yfeng1/m3127/highres/$bs-test-0.168/fastpm_$i/ --dataset=1 -- /global/cscratch1/sd/yfeng1/m3127/highres/$bs-test-0.168/fof_$i/ --dataset=LL-0.200
done
