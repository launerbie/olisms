#!/usr/bin/env python

import os
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import argparse
import progressbar as pb

# TODO: select only a subset of available Temperatures
# TODO: seperate arguments for enery acf figure and magnetization acf figure
# TODO: logscale

def get_basename(filepath):
    """ If filepath = 'some_path/myfile.hdf5', then this returns 'myfile'. """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    return name

def acf(ndarr, length=None):
    """
    Return an ndarray with normalized correlation coefficients corresponding
    with lag elements given by: range(1, length).
    The range starts with 1 because the correlation for lag=0 is infinity.

    Parameters
    ----------
    ndarr : ndarray of shape (1,)
        The signal (typically time-series data) for which the autocorrelation
        function C(lag) needs to calculated.

    length : int
        End of the interval range(1, length). The normalized correlation
        coefficients are calculated for this range and returned as an ndarray.
        #TODO: or optional array of lag elements [dt1, dt2, dt3,...]

    See:

    https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

    for the definition of the correlation matrix.

    The correlation coefficients are returned by np.corrcoef. See for details:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

    """
    if not length:
        length = len(ndarr) - 1

    coeffs = [np.corrcoef(ndarr[:-dt], ndarr[dt:])[0, 1] \
              for dt in range(1, length)]
    result = np.array([1]+ coeffs)
    return result


def make_acf_plot(h5pyfile, name, **kwargs):
    """
    Valid keyword arguments are:
    h5path:
    norminterval:
    cmap:
    length:
    xlim:
    img_suffix:
    targetdir:
    """

    if 'h5path' not in kwargs:
        raise Exception("Please specify timeseries data")
    else:
        h5path = kwargs['h5path']

    # Set up figure parameters here
    if 'norminterval' in kwargs:
        norm = colors.Normalize(*kwargs['norminterval'])
    else:
        norm = colors.Normalize(vmin=1.4, vmax=3.6)

    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = cm.hot

    if 'length' in kwargs:
        length = kwargs['length']
    else:
        length = None

    if 'xlim' in kwargs:
        xlim = kwargs['xlim']
    else:
        xlim = None

    if 'img_suffix' in kwargs:
        img_suffix = kwargs['img_suffix']
    else:
        img_suffix = "_{}_acf".format(h5path)

    if 'targetdir' in kwargs:
        targetdir = kwargs['targetdir']
    else:
        targetdir = 'figures'


    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    sims = h5pyfile.values()
    firstsim = list(h5pyfile.values())[0]
    shape = firstsim['shape'].value
    algorithm = h5pyfile.attrs['mode']

    shape_as_string = str(shape[0][0])+" x "+str(shape[0][1])
    # shape_as_string = "20 x 20"


    print("\nHDF5 file: {}".format(h5pyfile.filename))
    processs_description = "Generating ACF {}: ".format(h5path)
    pbar = pb.ProgressBar(widgets=drawwidget(processs_description),
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
                bbox_inches='tight')

    pbar.finish()

def drawwidget(discription):
    """ Formats the progressbar. """
    widget = [discription.ljust(20), pb.Percentage(), ' ',
              pb.Bar(marker='#', left='[', right=']'), ' ', pb.ETA()]
    return widget

def main():
    """
    Create figures in the target directory ARGS.targetdir.
    If it doesn't exist, it will be created.
    """
    from colors import rundark
    rundark()

    if not os.path.exists(ARGS.targetdir):
        os.makedirs(ARGS.targetdir)

    for hdf5file in ARGS.filenames:
        h5pyfile = h5py.File(hdf5file)
        name = get_basename(hdf5file)

        make_acf_plot(h5pyfile, name, h5path='magnetization', norm=ARGS.norm,
                      length=ARGS.length, xlim=ARGS.xlim)
        make_acf_plot(h5pyfile, name, h5path='energy', norm=ARGS.norm,
                      length=ARGS.length, xlim=ARGS.xlim)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs="+")
    parser.add_argument('--targetdir', default='figures')
    parser.add_argument('--xlim', nargs=2, metavar="xbegin xend", type=int)
    parser.add_argument('--norm', nargs=2, metavar="vmin vmax", type=float)
    parser.add_argument('--length', default=500, type=int)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    ARGS = get_arguments()
    main()

