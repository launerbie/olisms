#!/usr/bin/env python

import os
import h5py
import numpy as np
import numpy 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import argparse
import progressbar as pb

def main():
    """
    Create 'figures' directory if it doesn't exist. Make plots and save plots 
    to this directory.
    """
    from colors import rundark
    rundark()

    directory = 'figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = h5py.File(args.filename) 
    basename = os.path.basename(args.filename)
    name = os.path.splitext(basename)[0]

    make_combined_plots(f,directory,name)

def acf(x, length):
    """
    x: 1d array
    length: maximum dt for which to calculate correlation

    See https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    for the definition of the correlation matrix.
    """
    coeffs = [np.corrcoef(x[:-dt], x[dt:])[0,1] for dt in range(1,length)]
    return numpy.array([1]+ coeffs)

def make_combined_plots(f, directory, name):
    sims = f.values()

    #cmap = cm.RdBu_r 
    cmap = cm.hot 

    firstsim = list(f.values())[0]
    shape = firstsim['shape'][0]
    algorithm = f.attrs['mode']

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    pbar = pb.ProgressBar(widgets=drawwidget("Generating ACF Magnetization: "),
                  maxval=len(sims)).start()
    
    for simnr, sim in enumerate(sims):
        pbar.update(simnr)

        T = sim['temperature'][0]
        #temperature based linecolor
        norm = colors.Normalize(vmin=1.4,vmax=3.6)
        linecolor = cmap(norm(T))

        magnetization = sim['magnetization']

        ax1.plot(acf(magnetization, 500), c=linecolor)

    ax1.set_xlabel('dt [in sweeps]')
    ax1.set_ylabel('C (dt)')
    ax1.set_xlim(0,100)
    ax1.set_ylim(-0.1,1)
    ax1.set_title("{} {}".format(shape, algorithm))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbbar_handle = plt.colorbar(sm)
    cbbar_handle.set_label('Temperature')
    plt.savefig(directory+"/"+str(name)+"_mag_autocorr.png",
            bbox_inches='tight')

    pbar.finish()


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    pbar = pb.ProgressBar(widgets=drawwidget("Generating ACF Energy: "),
                  maxval=len(sims)).start()
    
    for simnr, sim in enumerate(sims):
        pbar.update(simnr)

        T = sim['temperature'][0]
        norm = colors.Normalize(vmin=1.4,vmax=3.6)
        linecolor = cmap(norm(T))

        energy = sim['energy']
        ax1.plot(acf(energy, 500), c=linecolor)

    ax1.set_xlabel('dt [in sweeps]')
    ax1.set_ylabel('C (dt)')
    ax1.set_xlim(0,100)
    ax1.set_ylim(-0.1,1)
    ax1.set_title("{} {}".format(shape, algorithm))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbbar_handle = plt.colorbar(sm)
    cbbar_handle.set_label('Temperature')
    plt.savefig(directory+"/"+str(name)+"_energy_autocorr.png",
            bbox_inches='tight')

    pbar.finish()


def drawwidget(discription):
    """ Formats the progressbar. """
    widgets = [discription.ljust(20), pb.Percentage(), ' ',
               pb.Bar(marker='#',left='[',right=']'),
               ' ', pb.ETA()]
    return widgets

    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar="HDF5 FILENAME")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()



