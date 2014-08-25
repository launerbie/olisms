#! /usr/bin/env python

import os
from scipy.optimize import minimize
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

def main():

    #syntax file = (hdf5-file, number of spins, lattice dimension)

    file1 = (h5py.File('/data/cursus3/wolff_data/wolff_20x20_10kiterations_50steps_2.hdf5'), 400, 2)
    file2 = (h5py.File('/data/cursus3/wolff_data/wolff_30x30_10kiterations_50steps.hdf5'), 900, 2)
    file3 = (h5py.File('/data/cursus3/wolff_data/wolff_40x40_10kiterations_50steps.hdf5'), 1600, 2)
    file4 = (h5py.File('/data/cursus3/wolff_data/wolff_50x50_10kiterations_50steps.hdf5'), 2500, 2)

    directory = args.directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = args.plotname



    Files = [file1, file2, file3, file4]

    true_gamma = minimize_chi_wiggle(Files)[0]
    true_nu = minimize_chi_wiggle(Files)[1]

    chi_values = []
    for f in Files:
        chi = generate_chi_array(f[0], f[1], f[2])
        chi_values.append(chi[0], chi[1], true_gamma, true_nu)

    plot_chi(chi_values)



def generate_chi_array(hdf5_file, number_of_spins, dimension):

    f = hdf5_file
    temperatures = []
    chi = []

    N = number_of_spins
    dim = float(dimension)
    L = np.power(N, 1/dim)

    for sim in hdf5_file.values():
        MCS0 = 500
        T = sim['temperature'][0]
        M = sim['magnetization'][-MCS0:]
        net_M = abs(M)

        temperatures.append(T)
        chi.append(1/(T*N)*np.var(net_M))
#        list_of_gridsize.append(N)

    return (np.array(chi), L)



def chi_wiggle(chi_array, L, gamma, nu):
    return np.power(L, -gamma/nu)*chi_array



def minimize_chi_wiggle(files):

    file1 = files[0]
    file2 = files[1]

    chi_1 = generate_chi_array(file1[0], file1[1], file1[2]) 
    chi_2 = generate_chi_array(file2[0], file2[1], file2[2])

    def square_diff_sum(x):
        diff = chi_wiggle(chi_1[0], chi_1[1], x[0], x[1]) -  chi_wiggle(chi_2[0], chi_2[1], x[0], x[1])
        diff_squared = np.power(diff, 2)
        return diff_squared.sum()

    x0 = [2, 1]

    x_min = minimize(square_diff_sum, x0)
    
    return x_min



def plot_chi(c_fits, directory, name):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for c in c_fits:
        label_string = str(int(np.sqrt(c[2][0]))) + 'x' + str(int(np.sqrt(c[2][0])))
        ax2.plot(c[0], c[3], 'o-', label=label_string)

        T.append(list(c[0]))
        C = C + list(c[3])

    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])


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
















    
