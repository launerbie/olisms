#!/usr/bin/env zsh

python runsim.py -x 40 -y 40   -a metropolis --tmin 2 --tmax 3 --steps 5 -i 32000000 --aligned -f data/metro_40x40_iter_32M_aligned.hdf5
python runsim.py -x 40 -y 40   -a metropolis --tmin 2 --tmax 3 --steps 5 -i 32000000 -f data/metro_40x40_iter_32M.hdf5

python runsim.py -x 20 -y 20   -a metropolis --tmin 1 --tmax 7 --steps 20 -i 8000000 -f data/metro_20x20_iter_8M.hdf5
python runsim.py -x 30 -y 30   -a metropolis --tmin 1 --tmax 7 --steps 20 -i 18000000 -f data/metro_30x30_iter_18M.hdf5
python runsim.py -x 40 -y 40   -a metropolis --tmin 1 --tmax 7 --steps 20 -i 32000000 -f data/metro_40x40_iter_32M.hdf5
python runsim.py -x 50 -y 50   -a metropolis --tmin 1 --tmax 7 --steps 20 -i 50000000 -f data/metro_50x50_iter_50M.hdf5
python runsim.py -x 60 -y 60   -a metropolis --tmin 1 --tmax 7 --steps 20 -i 72000000 -f data/metro_60x60_iter_72M.hdf5
python runsim.py -x 100 -y 100 -a metropolis --tmin 1 --tmax 7 --steps 20 -i 200000000 -f data/metro_100x100_iter_200M.hdf5


