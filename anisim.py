#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

from ising import IsingAnim

import threading
import time
from ext.colors import rundark

def main():
    if len(args.shape) == 2:
        animate_evolution()
    else:
        print("Only 2D lattices can be animated.")

def animate_evolution():
    plt.ion()
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    i = IsingAnim(args.shape, args.iterations, temperature=args.T, aligned=args.aligned,
              algorithm=args.algorithm)

    grid_2d = i.grid.reshape(args.shape[0], args.shape[1])

    if args.nointerpolate:
        im = ax.imshow(grid_2d, cmap=mpl.cm.binary, origin='lower', vmin=0,
                       vmax=1, interpolation='None' )
    else:
        im = ax.imshow(grid_2d, cmap=mpl.cm.binary, origin='lower', vmin=0,
                       vmax=1)

    def worker():
        i.evolve(sleep=args.s2)

    plt.draw()

    evolvegrid = threading.Thread(target=worker)
    evolvegrid.start()

    while evolvegrid.isAlive():
        time.sleep(args.s1)
        g = i.grid.reshape(args.shape[0], args.shape[1])
        im.set_array(g)
        ax.set_title(str(i.sweepcount))
        fig1.canvas.draw()


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--algorithm', choices=['metropolis','wolff'],
                        default='wolff')
    parser.add_argument('-i', '--iterations', default=1000000, type=int,
                        help="Number of iterations, default: 100000")
    parser.add_argument('--shape', default=[200, 200], type=int,
                        nargs='+', help="Lattice size")
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--nointerpolate', action='store_true')
    parser.add_argument('-T', default=2.3, type=float)
    parser.add_argument('--s1', default=0.1, type=float, help="image redraw interval")
    parser.add_argument('--s2', default=0.000001, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    rundark()
    main()
