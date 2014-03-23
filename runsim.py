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
    temperatures = numpy.linspace(0.0001, 1000, 2)

    with HDF5Handler(args.filename) as handler:
        simcount = 0
        for T in temperatures:
            print(T)
            sim_str = str(simcount).zfill(3)
            h5path = "/"+"sim_"+sim_str+"/"
            i = Ising(args.x, args.y, temperature=T, handler=handler, 
                      h5path=h5path)
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

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_arguments()
    main()
