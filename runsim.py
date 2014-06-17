#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy

from ising_model import Ising
from hdf5utils import HDF5Handler


def main():
    if os.path.exists(args.filename):
        print("{} already exists. Aborting.".format(args.filename))
        exit(0)
    else:
        simulate()
    

def simulate():
    """
    Run several Ising model simulations with different temperatures.
    Store them in 1 hdf5 file.

    The hdf5 file structure:

    /
    ├── sim_01
    │   ├── bfield
    │   ├── energy
    │   ├── iterations
    │   ├── magnetization
    │   ├── sites
    │   └── temperature
    ├── sim_02
    └── sim_03


    """
    Tmin = args.tmin
    Tmax = args.tmax
    steps = args.steps
    
    temperatures = numpy.linspace(Tmin, Tmax, steps)

    with HDF5Handler(args.filename) as handler:
        simcount = 0
        indexwidth = len(str(steps))
        for T in temperatures:
            print(T)
            sim_str = str(simcount).zfill(indexwidth)
            h5path = "/"+"sim_"+sim_str+"/"
            i = Ising(args.x, args.y, temperature=T, handler=handler, 
                      h5path=h5path, aligned=args.aligned)
            i.evolve(args.iterations) #TODO:need better stopping condition

            simcount += 1
            handler.file.flush()
        

def get_arguments():
    """
    To add arguments, call: parser.add_argument
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', choices=['metropolis','wolff'])
    parser.add_argument('-i', '--iterations', default=100000, type=int,
                        help="Number of iterations, default: 100000")
    parser.add_argument('-b', '--bfield', default=0.00, type=float,
                        help="Uniform external magnetic field, default: 0")
    parser.add_argument('-y', default=40, type=int, help="number of columns")
    parser.add_argument('-x', default=40, type=int, help="number of rows")
    parser.add_argument('-f', '--filename', required=True,
                        help="hdf5 output file name")

    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--tmin', default=0.001, type=float)
    parser.add_argument('--tmax', default=1000, type=float)
    parser.add_argument('--steps', default=5, type=float)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    main()
