#! /usr/bin/env python

import os
#from scipy.optimize import minimize
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from pprint import pprint
from misc import product

def main():

    #syntax file = (hdf5-file, number of spins, lattice dimension)

    file1 = h5py.File('/home/lau/scratch/oli/output/wolff_test/20x20_test_wolff_20x20_MCS45000_si20_minT1.8_maxT2.5_25_False.hdf5')
    file2 = h5py.File('/home/lau/scratch/oli/output/wolff_test/30x30_test_wolff_30x30_MCS45000_si20_minT1.8_maxT2.5_25_False.hdf5')
    file3 = h5py.File('/home/lau/scratch/oli/output/wolff_test/40x40_test_wolff_40x40_MCS45000_si20_minT1.8_maxT2.5_25_False.hdf5')
    file4 = h5py.File('/home/lau/scratch/ising_data/wolff_avg_2000_wolff_50x50_MCS25000_si20_minT1.85_maxT2.15_25_False.hdf5')
    file5 = h5py.File('/home/lau/scratch/ising_data/wolff_avg_2000_wolff_60x60_MCS25000_si20_minT1.85_maxT2.15_25_False.hdf5')

    directory = args.directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = args.plotname

    Files = [file1, file2, file3, file4, file5]

#    true_gamma = minimize_chi_wiggle(Files)[0]
#    true_nu = minimize_chi_wiggle(Files)[1]

    true_gamma = 1.76
    true_nu = 1.00


    chi_values = []
    for f in Files:
        chi = generate_chi_array(f, true_gamma, true_nu)
        chi_values.append(chi)

    plot_chi(chi_values, directory, name)



#def generate_chi_array(hdf5_file, number_of_spins, dimension):
#
#    f = hdf5_file
#    temperatures = []
#    chi = []
#
#    N = number_of_spins
#    dim = float(dimension)
#    L = np.power(N, 1/dim)
#
#    for sim in hdf5_file.values():
#        MCS0 = 500
#        T = sim['temperature'][0]
#        M = sim['magnetization'][-MCS0:]
#        net_M = abs(M)
#
#        temperatures.append(T)
#        chi.append(1/(T*N)*np.var(net_M))
##        list_of_gridsize.append(N)
#
#    return (np.array(chi), L)



def lattice_size_from_shape(shape):
    """
    if shape = '40x40', then return 1600
    """
    dimensions = [int(i) for i in shape.split('x')]
    return product(dimensions)



def generate_chi_array(f, gamma, nu):
#    fig = plt.figure(figsize = (8,4))
#    ax1 = fig.add_subplot(111)
#
#    plt.rc('text', usetex=True)
#    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]


    temperatures = []
    avg_net_mags = []
    chi = []

    pprint(list(f.attrs.items()))
    N = lattice_size_from_shape(f.attrs['shape'])
    L = np.sqrt(N)


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

        temperatures.append(T)
        chi.append(1/(T*N)*np.var(net_M))

    betas = 1.0/np.array(temperatures)[::-1]
    reversed_chi = np.array(chi)[::-1]



    chi_to_plot = chi_wiggle(reversed_chi, L, gamma, nu)

    return (betas, chi_to_plot, L)




def chi_wiggle(chi_array, L, gamma, nu):
    return np.power(L, -gamma/nu)*chi_array



#def minimize_chi_wiggle(files):
#
#    file1 = files[0]
#    file2 = files[1]
#
#    chi_1 = generate_chi_array(file1[0], file1[1], file1[2]) 
#    chi_2 = generate_chi_array(file2[0], file2[1], file2[2])
#
#    def square_diff_sum(x):
#        diff = chi_wiggle(chi_1[0], chi_1[1], x[0], x[1]) -  chi_wiggle(chi_2[0], chi_2[1], x[0], x[1])
#        diff_squared = np.power(diff, 2)
#        return diff_squared.sum()
#
#    x0 = [2, 1]
#
#    x_min = minimize(square_diff_sum, x0)
#    
#    return x_min



def plot_chi(c_fits, directory, name):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for beta, chi, gridsize in c_fits:
        label_string = "L = " + str(int(gridsize))
        ax2.plot(beta, chi, 'o-', label=label_string)

#        T.append(list(c[0]))
#        C = C + list(c[1])

#    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])


    ax2.set_xlabel('Temperatures')
    ax2.set_ylabel('Magnetic susceptibility')
    ax2.legend(loc='best', prop={'size':12})
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    plt.savefig(directory+"/"+str(name)+"_data_collapse"+".png", bbox_inches='tight')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('plotname')
    parser.add_argument('-d', '--directory', default='data_collapse', type=str)


    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
















    
