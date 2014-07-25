#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy

from ising import Ising
from hdf5utils import HDF5Handler


def main():
    if os.path.exists(args.filename):
        pythonversion = sys.version_info[0]

        key = None
        while key not in ['y', 'n']:

            if pythonversion == 2:
                key = raw_input("{} exists. Overwrite? y/n: ".format(args.filename))
            else:
                key = input("{} exists. Overwrite? y/n: ".format(args.filename))

            if key == 'y':
                os.remove(args.filename)
                simulate()
            elif key == 'n':
                print("Aborting runsim.py")
                exit(0)
            else:
                print("Please input 'y' or 'n'\n")
    else:
        simulate()
    

def simulate():
    """
    Run several Ising model simulations with different temperatures.
    Store them in 1 hdf5 file.

    """
    temperatures = numpy.linspace(args.tmin, args.tmax, args.steps)

    with HDF5Handler(args.filename) as handler:
        simcount = 0
        for T in temperatures:
            print(T)
            sim_str = str(simcount).zfill(4)
            h5path = "/"+"sim_"+sim_str+"/"
            i = Ising(args.shape, temperature=T, handler=handler, 
                      h5path=h5path, aligned=args.aligned, mode=args.algorithm,
                      saveinterval=args.saveinterval)
            i.evolve(args.iterations) 

            simcount += 1
            handler.file.flush()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', help="hdf5 output file name", required=True)
    parser.add_argument('-a', '--algorithm', choices=['metropolis','wolff'], required=True)
    parser.add_argument('-i', '--iterations', default=100000, type=int,
                        help="Number of iterations, default: 100000")
    parser.add_argument('--shape', default=[40, 40], type=int, 
                        nargs='+', help="Lattice size")
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--tmin', default=0.1, type=float)
    parser.add_argument('--tmax', default=10, type=float)
    parser.add_argument('--steps', default=5, type=float)
    parser.add_argument('--saveinterval', default=10000, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    main()
