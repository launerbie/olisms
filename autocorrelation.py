#!/usr/bin/env python

import os
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import ext.progressbar as progressbar
from ext.colors import rundark
from ext.colors import runbright
from misc import drawwidget
from misc import get_basename
from misc import acf

# TODO: select only a subset of available Temperatures
# TODO: seperate arguments for enery acf figure and magnetization acf figure
# TODO: logscale

def make_acf_plot(h5pyfile, name, **kwargs):
    """
    Parameters
    ----------
    Valid keyword arguments are:

    h5path : str
        Pass an h5path to specify which data you want use, typically h5path
        is either "energy" or "magnetization". Since ['sim_0000']['energy'])
        or ['sim_0000']['magnetization'] are timeseries data for which you
        want to calculate the time autocorrelation function.

    norminterval: (float, float). Default: (1.4, 1.6)
        Used to normalize the color range between (vmin, vmax).

    cmap: A matplotlib colormap. Default: matplotlib.cm.hot

    length: int
        The maximum time lag. Used as range(1, length), thereby calculating
        (length - 1) correlation coefficients.

    xlim: (int, int)
        The x axes limits.

    img_suffix: str

    targetdir: str, default = 'figures'

    """

    if 'h5path' not in kwargs:
        raise Exception("Please specify timeseries data")
    else:
        h5path = kwargs['h5path']

    # Set up figure parameters here
    if 'norminterval' in kwargs:
        norm = mpl.colors.Normalize(*kwargs['norminterval'])
    else:
        norm = mpl.colors.Normalize(vmin=1.4, vmax=3.6)

    cmap = kwargs.get('cmap', cm.hot)
    length = kwargs.get('length', 500)
    dpi = kwargs.get('dpi', 80)
    xlim = kwargs.get('xlim', None)
    img_suffix = kwargs.get('img_suffix', '_{}_acf'.format(h5path))
    targetdir = kwargs.get('targetdir', 'figures')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    sims = h5pyfile.values()
    firstsim = list(h5pyfile.values())[0]
    shape = firstsim['shape'].value
    algorithm = h5pyfile.attrs['algorithm']

    shape_as_string = str(shape[0][0])+" x "+str(shape[0][1])
    # shape_as_string = "20 x 20"

    print("\nHDF5 file: {}".format(h5pyfile.filename))
    processs_description = "Generating ACF {}: ".format(h5path)
    pbar = progressbar.ProgressBar(widgets=drawwidget(processs_description),
                                   maxval=len(sims)).start()

    for simnr, sim in enumerate(sims):
        pbar.update(simnr)
        temperature = sim['temperature'][0]
        timeseriesdata = sim[h5path]
        ax1.plot(acf(timeseriesdata, length), c=cmap(norm(temperature)))

    if algorithm == 'metropolis':
        ax1.set_xlabel('lag [in sweeps ]')

    elif algorithm == 'wolff':
        ax1.set_xlabel('lag [in clusterflips]')

    else:
        ax1.set_xlabel('lag [in ??]')

    ax1.set_ylabel('C (lag)')

    if xlim:
        ax1.set_xlim(*xlim)

    ax1.set_ylim(-0.1, 1)

    substitutions = (h5path, shape_as_string, algorithm)
    ax1.set_title("Time series data: {} \n{} {}".format(*substitutions))

    scalarmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalarmap._A = [] #How to get around this?
    cbbar_handle = plt.colorbar(scalarmap)
    cbbar_handle.set_label('Temperature')

    plt.savefig(targetdir + "/" + name + img_suffix + ".png",
                bbox_inches='tight', dpi=dpi)

    pbar.finish()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs="+")
    parser.add_argument('--targetdir', default='figures')
    parser.add_argument('--xlim', nargs=2, metavar="xbegin xend", type=int)
    parser.add_argument('--norm', nargs=2, metavar="vmin vmax", type=float)
    parser.add_argument('--length', default=500, type=int)
    parser.add_argument('--runbright', action="store_true")
    parser.add_argument('--dpi', default=80, type=int)
    arguments = parser.parse_args()
    return arguments

def main():
    """ Create figures in the target directory ARGS.targetdir. """
    if ARGS.runbright:
        runbright()
    else:
        rundark()

    if not os.path.exists(ARGS.targetdir):
        os.makedirs(ARGS.targetdir)

    for hdf5file in ARGS.filenames:
        h5pyfile = h5py.File(hdf5file)

        name = get_basename(hdf5file)
        # get_basename('path_to/myfile.hdf5') = 'myfile'

        make_acf_plot(h5pyfile, name, h5path='magnetization', norm=ARGS.norm,
                      length=ARGS.length, xlim=ARGS.xlim, dpi=ARGS.dpi)
        make_acf_plot(h5pyfile, name, h5path='energy', norm=ARGS.norm,
                      length=ARGS.length, xlim=ARGS.xlim, dpi=ARGS.dpi)


if __name__ == "__main__":
    ARGS = get_arguments()
    main()
