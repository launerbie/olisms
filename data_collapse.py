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
    offset = args.offset

    Files = [file1, file2, file3, file4]

    true_gamma = minimize_chi_wiggle(Files, offset)[0]
    true_nu = minimize_chi_wiggle(Files, offset)[1]

#    true_gamma = 1.75
#    true_nu = 1

    chi_values = []
    for f in Files:
        chi = generate_chi_array(f[0], f[1], f[2])
        chi_wig = chi_wiggle(chi[0], chi[1], chi[2], true_gamma, true_nu)
        chi_values.append(chi_wig)

    plot_chi(chi_values, directory, name)

    print true_gamma
    print true_nu


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

    return (np.array(chi), L, temperatures)



def chi_wiggle(chi_array, L, temps, gamma, nu):

    chi_wig = np.power(L, -gamma/nu)*chi_array

    return (chi_wig, temps)


def minimize_chi_wiggle(files, offset):

    file1 = files[0]
    file1_chi_array = generate_chi_array(file1[0], file1[1], file1[2])

    length_chi = len(file1_chi_array[2])
    ub = int(length_chi/2) + offset #Upper bound
    lb = int(length_chi/2) - offset #Lower bound
 
    x0 = [1.75, 1]

    gamma_list = []
    nu_list = []

    buff = file1

    for i, f in enumerate(files):
        if i == 1:
            pass
 
        else:

            chi_1 = generate_chi_array(f[0], f[1], f[2]) 
            chi_2 = generate_chi_array(buff[0], buff[1], buff[2])
            buff = f

            def square_diff_sum(x):
                diff = chi_wiggle(chi_1[0][lb:ub], chi_1[1], chi_1[2][lb:ub], x[0], x[1])[0] - chi_wiggle(chi_2[0][lb:ub], chi_2[1], chi_2[2][lb:ub], x[0], x[1])[0]
                diff_squared = np.power(diff, 2)
                return diff_squared.sum()

            x_min = minimize(square_diff_sum, x0)['x']
            gamma_list.append(x_min[0])
            nu_list.append(x_min[1])

                   
#    chi_1 = generate_chi_array(file1[0], file1[1], file1[2]) 
#    chi_2 = generate_chi_array(file2[0], file2[1], file2[2])

    gamma_mean = np.array(gamma_list).mean()
    nu_mean = np.array(nu_list).mean()
        
    return (gamma_mean, nu_mean)



def plot_chi(c_fits, directory, name):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for c in c_fits:
#        label_string = str(int(np.sqrt(c[2][0]))) + 'x' + str(int(np.sqrt(c[2][0])))
        ax2.plot(c[1], c[0], 'o-') #, label=label_string)

        T.append(list(c[1]))
        C = C + list(c[0])

    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])


    ax2.set_xlabel('Temperatures')
    ax2.set_ylabel('Magnetic susceptibility')
#    ax2.legend(loc='best', prop={'size':12})
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    plt.savefig(directory+"/"+str(name)+"_data_collapse"+".png", bbox_inches='tight')



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('plotname')
    parser.add_argument('-d', '--directory', default='data_collapse', type=str)
    parser.add_argument('-o', '--offset', default=5, type=int)


    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
















    
