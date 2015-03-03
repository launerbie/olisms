#!/usr/bin/env python

import os
import h5py
import numpy
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import matplotlib.gridspec as gridspec
from pprint import pprint
from misc import product

"""
Zie:
http://scipy-lectures.github.io/_images/plot_colormaps_1.png
voor een plaatje met deze colormaps.
"""

def main():
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
    fig = plt.figure(figsize=(8,10))
    gs = gridspec.GridSpec(3, 2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])
    ax4 = fig.add_subplot(gs[2,:])

    cmap = cm.jet

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

        #temperature based linecolor
        norm = colors.Normalize(vmin=2.0,vmax=3.5)
        linecolor = cmap(norm(T))

        ax3.plot(time, E, color=linecolor)
        ax4.plot(time, net_M, color=linecolor)

    ax1.plot(temperatures, numpy.array(avg_net_mags)/N, c='k', marker='o', ms=4)
    ax2.plot(numpy.array(temperatures), chi, c='k', marker='o', ms=4)

    ax1.set_xlabel('T')
    ax1.set_ylabel('<|M|>')
    ax2.set_xlabel('T')
    ax2.set_ylabel('Chi')

    if algorithm == 'wolff':
        ax3.set_xlabel('time [clusterflips]')
        ax4.set_xlabel('time [clusterflips]')
    elif algorithm == 'metropolis':
        ax3.set_xlabel('time [sweeps]')
        ax4.set_xlabel('time [sweeps]')

    ax3.set_ylabel('E')
    ax4.set_ylabel('|M|')

    if args.xlim is not None:
        ax3.set_xlim(*args.xlim)
        ax4.set_xlim(*args.xlim)

    maps = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax, kw = mpl.colorbar.make_axes([ax3,ax4])
    maps._A = [] #TODO: remove ugly hack
    plt.colorbar(maps, cax=cax)

    fig.suptitle('{}\n'.format(f.filename), fontsize=12)

    if not args.plot:
        plt.savefig(directory+"/"+str(name)+"_summary"+".png", bbox_inches='tight')
        fig.clf()
        plt.close()
    else:
        plt.show()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('--outputdir', default='figures')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--xlim', nargs=2, default=None, metavar="xbegin xend", type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
