#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

from ising import Ising

import threading
import time
from colors import rundark

def main():
    if len(args.shape) == 2:
        animate_evolution()
    else:
        print("Only 2D lattices can be animated.") 

def animate_evolution():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = Ising(args.shape, temperature=args.T, aligned=args.aligned, 
              mode=args.algorithm)

    if args.nointerpolate:
        im = ax.imshow(i.grid, cmap=mpl.cm.binary, origin='lower', vmin=-1, 
                       vmax=1, interpolation='None' )
    else:
        im = ax.imshow(i.grid, cmap=mpl.cm.binary, origin='lower', vmin=-1,
                       vmax=1)

    def worker():
        i.evolve(args.iterations, sleep=args.s2)

    plt.draw()
 
    evolvegrid = threading.Thread(target=worker)
    evolvegrid.start()

    while evolvegrid.isAlive():
        time.sleep(args.s1)
        im.set_array(i.grid)
        ax.set_title(str(i.i))
        plt.draw()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', choices=['metropolis','wolff'],
                        default='metropolis')
    parser.add_argument('-i', '--iterations', default=1e6, type=int,
                        help="Number of iterations, default: 100000")
    parser.add_argument('--shape', default=[200, 200], type=int, 
                        nargs='+', help="Lattice size")
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--nointerpolate', action='store_true')
    parser.add_argument('-T', default=1.5, type=float)
    parser.add_argument('--s1', default=0.1, type=float, help="image redraw interval")
    parser.add_argument('--s2', default=0.000001, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    main()
