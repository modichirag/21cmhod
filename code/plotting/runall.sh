#!/bin/bash

python plot_pkbk.py -m ModelA -s small
python plot_pkbk.py -m ModelB -s small
python plot_pkbk.py -m ModelC -s small

python plot_pkbk.py -m ModelA -s big
python plot_pkbk.py -m ModelB -s big
python plot_pkbk.py -m ModelC -s big


python plot_bao_cm.py -m ModelA -s small
python plot_bao_cm.py -m ModelB -s small
python plot_bao_cm.py -m ModelC -s small

python plot_bao_cm.py -m ModelA -s big
python plot_bao_cm.py -m ModelB -s big
python plot_bao_cm.py -m ModelC -s big
