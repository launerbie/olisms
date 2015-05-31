#!/usr/bin/env python

import os
import h5py
import numpy
import argparse
from pprint import pprint
from misc import product

"""
Zie:
http://scipy-lectures.github.io/_images/plot_colormaps_1.png
voor een plaatje met deze colormaps.
"""

def main():
    print("backend:", mpl.get_backend())
    directory = args.outputdir
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in args.filenames:
        f = h5py.File(filename)
        basename = os.path.basename(filename)
        name = os.path.splitext(basename)[0]

        plot_summary(f, name, directory)

def lattice_size_from_shape(shape):
    """
    if shape = '40x40', then return 1600
    """
    dimensions = [int(i) for i in shape.split('x')]
    return product(dimensions)

def plot_summary(f, name, directory):
    fig = plt.figure(figsize = (15,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]



    temperatures = []
    avg_net_mags = []
    chi = []

    pprint(list(f.attrs.items()))
    N = lattice_size_from_shape(f.attrs['shape'])

    try:
        commit = str(f.attrs['commit'])
        print("Created with commit:", commit)
    except KeyError:
        print("Warning: 'commit' not stored in root.attrs")


    algorithm = f.attrs['algorithm']

    for sim in f.values(): #Each sim corresponds to a simulation at some Temperature

        if algorithm == 'wolff':
            time = sim['clusterflip']
        else:
            time = sim['sweep']

        T = sim['temperature'][0]
        M = sim['magnetization'].value
        net_M = abs(M)
        E = sim['energy'].value

        temperatures.append(T)
        avg_net_mags.append(numpy.mean(net_M))
        chi.append(1/(T*N)*numpy.var(net_M))

    betas = 1.0/numpy.array(temperatures)[::-1]
#    betas = numpy.around(1.0/numpy.array(temperatures)[::-1], decimals = 2)
    reversed_net_mags = numpy.array(avg_net_mags)[::-1]
    reversed_chi = numpy.array(chi)[::-1]

    ax1.plot(betas, reversed_net_mags/N, mfc='k', c = 'k', marker='o', ms=4)
    ax2.plot(betas, reversed_chi, mfc='k', c = 'k', marker='o', ms=4)

    ax1.set_xlabel(r"$\boldsymbol\beta$", fontsize=15)
    ax1.set_ylabel(r"$\boldsymbol{\langle | M | \rangle / N}$")
    ax2.set_xlabel(r"$\boldsymbol\beta$", fontsize=15)
    ax2.set_ylabel(r"$\boldsymbol\chi$", fontsize=15)

    ax1.tick_params(
        axis = 'x',
        which = 'both',
        bottom = 'off',
        top = 'off')

    ax1.tick_params(
        axis = 'y',
        which = 'both',
        left = 'off',
        right = 'off')

    ax2.tick_params(
        axis = 'x',
        which = 'both',
        bottom = 'off',
        top = 'off')

    ax2.tick_params(
        axis = 'y',
        which = 'both',
        left = 'off',
        right = 'off')


    if args.plot is False:
        plt.savefig(directory+"/"+str(name)+"_summary"+".png", bbox_inches='tight')
        fig.clf()
        plt.close()
    else:
        plt.show()
    #plt.show()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('--outputdir', default='figures')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--xlim', nargs=2, default=None, metavar="xbegin xend", type=int)
    parser.add_argument('--bounds', nargs=2, default=[2.0, 3.0], metavar="vmin vmax", type=float)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)

    import matplotlib as mpl

    if args.plot is True:
        mpl.use('TkAgg')
    else:
        mpl.use('Agg')

    import matplotlib.pyplot as plt
#    from matplotlib import cm
#    from matplotlib import colors
#    import matplotlib.gridspec as gridspec

    main()
