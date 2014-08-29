#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy
from ising import Ising

from ext import progressbar as pb
from misc import drawwidget
from ext.hdf5handler import HDF5Handler


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
        for index, T in enumerate(temperatures):
            h5path = "/"+"sim_"+str(index).zfill(4)+"/"
            # h5path thus looks like:
            # "/sim_0000/", "/sim_0001/", etc.

            i = Ising(args.shape, args.sweeps, temperature=T, handler=handler,
                      h5path=h5path, aligned=args.aligned, mode=args.algorithm,
                      saveinterval=args.saveinterval, skip_n_steps=args.skip)

            if args.verbose:
                i.print_sim_parameters()

            widgets=drawwidget("  T = {}  {}/{} ".format(round(T,4), index+1,
                                                         len(temperatures)))
            pbar = pb.ProgressBar(widgets=widgets, maxval=args.sweeps).start()
            i.evolve(pbar)
            pbar.finish()

            handler.file.flush()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--filename', help="hdf5 output file name", required=True)
    parser.add_argument('-a', '--algorithm',
                        choices=['metropolis','wolff'],
                        default='metropolis')
    parser.add_argument('-s', '--sweeps', default=20000, type=int,
                        help="Number of sweeps, default: 20000")
    parser.add_argument('--shape', default=[20, 20], type=int,
                        nargs='+', help="Lattice size")
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--tmin', default=1.5, type=float)
    parser.add_argument('--tmax', default=3.5, type=float)
    parser.add_argument('--steps', default=10, type=float)
    parser.add_argument('--saveinterval', default=50, type=int)
    parser.add_argument('--skip', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    main()
