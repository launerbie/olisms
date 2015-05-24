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

        plot(f, name, directory)

def lattice_size_from_shape(shape):
    """
    if shape = '40x40', then return 1600
    """
    dimensions = [int(i) for i in shape.split('x')]
    return product(dimensions)

def plot(f, name, directory):

    fig = plt.figure(figsize = (15,4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    N = lattice_size_from_shape(f.attrs['shape'])
    algorithm = f.attrs['algorithm']

    for sim in f.values(): #Each sim corresponds to a simulation at some Temperature
      
        if algorithm == 'wolff':
            time = sim['clusterflip']
        else:
            time = sim['sweep']
 
        T = sim['temperature'][0]

        E = sim['energy'].value

        M = sim['magnetization'].value
        net_M = abs(M)

        beta = str(round(1./T, 2))
        legend_entry = r"$\beta$ = " + beta

        ax1.plot(time, E/N, label= legend_entry)
        ax2.plot(time, net_M/N)
       
    if algorithm == 'metropolis':
        ax1.set_xlabel('time [sweeps]')
        ax2.set_xlabel('time [sweeps]')
    elif algorithm == 'wolff':
        ax1.set_xlabel('time [clusterflips]') 
        ax2.set_xlabel('time [clusterflips]') 


    ax1.set_ylabel('Energy')
    ax2.set_ylabel('Net magnetization')
    ax1.legend(loc='upper center', bbox_to_anchor=(1, 1.22), ncol=5)

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
        plt.savefig(directory+"/"+str(name)+"_Energy_and_Magnetization"+".png", bbox_inches='tight')
        fig.clf()
        plt.close()
    else:
        plt.show()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+')
    parser.add_argument('--outputdir', default='figures')
    parser.add_argument('--plot', action='store_true')
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

    main()


    
