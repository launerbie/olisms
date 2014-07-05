#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy

from ising import Ising
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

    """
    temperatures = numpy.linspace(args.tmin, args.tmax, args.steps)

    with HDF5Handler(args.filename) as handler:
        simcount = 0
        for T in temperatures:
            print(T)
            sim_str = str(simcount).zfill(4)
            h5path = "/"+"sim_"+sim_str+"/"
            i = Ising(args.shape, temperature=T, handler=handler, 
                      h5path=h5path, aligned=args.aligned, mode=args.algorithm)
            i.evolve(args.iterations) 

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
    parser.add_argument('-f', '--filename', required=True,
                        help="hdf5 output file name")

    parser.add_argument('--shape', default=[40, 40], type=int, nargs='+', help="Lattice size")
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--tmin', default=0.1, type=float)
    parser.add_argument('--tmax', default=10, type=float)
    parser.add_argument('--steps', default=5, type=float)

    #if len(args.shape) < 2, abort

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
