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

"""
De beschikbare colormaps zijn:

'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 
'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 
'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 
'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 
'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 
'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 
'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'YlGn', 
'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 
'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cool', 'cool_r', 'coolwarm', 
'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 
'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 
'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 
'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 
'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 
'hsv_r', 'jet', 'jet_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
'pink', 'pink_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 
'seismic_r', 'spectral', 'spectral_r', 'spring', 'spring_r', 'summer', 
'summer_r', 'terrain', 'terrain_r', 'winter', 'winter_r']

dus matplotlib.cm.Accent is bijvoorbeeld een colormap.

Zie:
http://scipy-lectures.github.io/_images/plot_colormaps_1.png
voor een plaatje met deze colormaps.

"""

def main():
    directory = 'figures'
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = h5py.File(args.filename) 
    basename = os.path.basename(args.filename)
    name = os.path.splitext(basename)[0]

    def plot_summary():
        fig = plt.figure(figsize=(8,10))        
        gs = gridspec.GridSpec(3, 2)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,:])
        ax4 = fig.add_subplot(gs[2,:])

        cmap = cm.RdBu #red blue
        
        temperatures = []
        avg_net_mags = []
        chi = []

        firstsim = list(f.values())[0]

        shape = firstsim['shape'][0]
        saveinterval = firstsim['saveinterval'][0]
        algorithm = 'unknown' # get from hdf5...
        N = firstsim['lattice_size'][0]  
        MCS0 = int(2*saveinterval) #cut-off point

        for sim in f.values(): #Each sim corresponds to a simulation at some Temperature
            T = sim['temperature'][0]
            M = sim['magnetization'][-MCS0:]
            net_M = abs(M)
            E = sim['energy'][-MCS0:]
            sweep = sim['sweep'][-MCS0:]

            temperatures.append(T)
            avg_net_mags.append(numpy.mean(net_M))
            chi.append(1/(T*N)*numpy.var(net_M))

            #temperature based linecolor
            norm = colors.Normalize(vmin=1.8,vmax=3.2)
            linecolor = cmap(norm(T))

            ax3.plot(sweep, E, color=linecolor)
            ax4.plot(sweep, net_M, color=linecolor)

        ax1.plot(temperatures, numpy.array(avg_net_mags)/N, c='k', marker='o', ms=4)
        ax2.plot(numpy.array(temperatures), chi, c='k', marker='o', ms=4)

        ax1.set_xlabel('T')
        ax1.set_ylabel('<|M|>')
        ax2.set_xlabel('T')
        ax2.set_ylabel('Chi')
        ax3.set_xlabel('time [sweeps]')
        ax3.set_ylabel('E')
        ax4.set_xlabel('time [sweeps]')
        ax4.set_ylabel('|M|')



        maps = mpl.cm.ScalarMappable(norm=norm, cmap=cm.RdBu)
        cax, kw = mpl.colorbar.make_axes([ax3,ax4])
        maps._A = [] #TODO: remove ugly hack
        plt.colorbar(maps, cax=cax)

        fig.suptitle('{}   {} '.format(f.filename, shape) ,fontsize=12)
        #TODO: include algorithm in title

        if not args.plot:
            plt.savefig(directory+"/"+str(name)+"_summary"+".png", bbox_inches='tight')
            fig.clf()
            plt.close()
        else:
            plt.show()

    plot_summary()

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar="HDF5 FILENAME")
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
