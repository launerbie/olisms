#! /usr/bin/env python

import os
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt

'''
Geschaalde magnetische susceptibiliteit is gedefineerd als: chi_wiggle = L^(-gamma/nu)*chi. 
Hier is L het aantal spins in 1 dimensie, dus totaal aantal spins = L^dimensie. Zie voor
afleiding paragraaf 8.3 van Barkema and Newman. Voor geschikte waardes van gamma en nu zijn,
idealiter, de curvers chi_wiggle voor elke waarde van L hetzelfde, de zogenaamde "data collapse".
Dit programma zoekt waardes van gamma en nu waarvoor het verschil tussen de maxima van chi_wiggle
van elk opeenvolgend paar input chi-arrays minimaal is, en middelt vervolgens over al die waardes 
om zo tot de "goede" gamma en nu te komen. Een plot van de chi_wiggle's, met de gevonden waardes
van gamma en nu, van alle input data-sets wordt gemaakt in de map 'data_collapse'.
-------------------------------------------------------------------------------------------------
'''


def main():

    #syntax file = (hdf5-file, number of spins, lattice dimension)

    file1 = (h5py.File('/data/cursus3/ising_data_3/mtr_20x20_50steps_S20000_si10.hdf5'), 400, 2)
    file2 = (h5py.File('/data/cursus3/ising_data_3/mtr_30x30_50steps_S20000_si10.hdf5'), 900, 2)
    file3 = (h5py.File('/data/cursus3/ising_data_3/mtr_40x40_50steps_S20000_si10.hdf5'), 1600, 2)
    file4 = (h5py.File('/data/cursus3/ising_data_3/mtr_50x50_50steps_S20000_si10.hdf5'), 2500, 2)
    file5 = (h5py.File('/data/cursus3/ising_data_3/mtr_60x60_50steps_S20000_si10.hdf5'), 3600, 2)


#    file1 = (h5py.File('/data/cursus3/wolff_data/wolff_20x20_10kiterations_50steps_2.hdf5'), 400, 2)
#    file2 = (h5py.File('/data/cursus3/wolff_data/wolff_30x30_10kiterations_50steps.hdf5'), 900, 2)
#    file3 = (h5py.File('/data/cursus3/wolff_data/wolff_40x40_10kiterations_50steps.hdf5'), 1600, 2)
#    file4 = (h5py.File('/data/cursus3/wolff_data/wolff_50x50_10kiterations_50steps.hdf5'), 2500, 2)


#    file1 = (h5py.File('/data/cursus3/wolff_3D_data/8x8x8_T3-5_steps50_sv1_sweeps2k.hdf5'), 512, 3)
#    file2 = (h5py.File('/data/cursus3/wolff_3D_data/10x10x10_T3-5_steps50_sv1_sweeps2k.hdf5'), 1000, 3)
#    file3 = (h5py.File('/data/cursus3/wolff_3D_data/12x12x12_T3-5_steps50_sv1_sweeps2k.hdf5'), 1728, 3)
#    file4 = (h5py.File('/data/cursus3/wolff_3D_data/16x16x16_T3-5_steps50_sv1_sweeps2k.hdf5'), 2744, 3)


    directory = args.directory # Default = data_collapse
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = args.plotname
    offs = args.offset

    Files = [file1, file2, file3, file4, file5]

    mins = minimize_chi_wiggle(Files, offs)
   
    true_gamma = mins[0]
    true_nu = mins[1]


#    true_gamma = 1.75  # Uncomment deze twee regels om de juiste waardes van gamma en nu 
#    true_nu = 1        # te gebruiken (voor 2D zijn dit de theoretische waardes)


    # Onderstaande for-loop herschaalt de data na het vinden van de "goede" exponenten. 
    chi_values = []
    for f in Files:
        chi = generate_chi_array(f)
        chi_wig = chi_wiggle(chi[0], chi[1], chi[2], true_gamma, true_nu)
        chi_values.append(chi_wig)

    plot_chi(chi_values, directory, name, offs, true_gamma, true_nu)

    print true_gamma
    print true_nu



def fit_gauss(x_range, data):

    '''
    Om gebruik te kunnen maken van de fits, in de hoop dat dat misschien betere resultaten 
    geeft. 
    --------------------------------------------------------------------------------------
    '''

    def func(x, a, b, c, d):
        return a * np.exp(-((x - b)**2)/(2 * c**2)) + d

    popt, pcov = curve_fit(func, x_range, data)

    return func(x_range, popt[0], popt[1], popt[2], popt[3])




def generate_chi_array(file_tuple):

    f = file_tuple[0] #hdf5_file

    temperatures = []
    chi = []

    N = file_tuple[1] # number_of_spins
    dim = float(file_tuple[2]) # dimension
    L = np.power(N, 1/dim)

    for sim in f.values():
        MCS0 = 800
        T = sim['temperature'][0]
        M = sim['magnetization'][-MCS0:]
        net_M = abs(M)

        temperatures.append(T)
        chi.append(1/(T*N)*np.var(net_M))


    fit = fit_gauss(np.array(temperatures), chi)
    return (np.array(chi), L, temperatures, fit)



def chi_wiggle(chi_array, L, temps, gamma, nu):

    chi_wig = np.power(L, -gamma/nu)*chi_array

    return (chi_wig, temps)



def scaling_factor(data, L, gamma, nu):
    return np.power(L, -gamma/nu)*data
    # Misschien nuttig om dit als aparte
    # functie te houden, maar eigenlijk
    # overbodig in dit programma.


def min_sqr_diff(input1, L1, input2, L2):
    '''
    Minimaliseert (L1*(-gamma/nu)*input1 - L2^(-gamma/nu)*input2)^2 door variatie van 
    gamma en nu. Output: (gamma_optimaal, nu_otimaal). De inputs mogen een getal of een
    numpy array zijn.
    -----------------------------------------------------------------------------------
    '''

    type1 = type(input1)

    if type1 == type(input2):

        def sqr_diff(x):
            diff = scaling_factor(input1, L1, x[0], x[1]) - scaling_factor(input2, L2, x[0], x[1])
            diff_squared = np.power(diff, 2)
            if type1 == np.ndarray:
                return diff_squared.sum()
            elif type1 == np.float64 or type1 == int or type1 == float:
                return diff_squared
            else:
                s = "Input data must be number or array. (" + '{}'.format(str(type1)) + ")"
                raise TypeError(s)

        def func(x):
            diff = scaling_factor(input1, L1, x[0], x[1]) - scaling_factor(input2, L2, x[0], x[1])
            if type1 == np.ndarray:
                return diff.sum()
            elif type1 == np.float64 or type1 == int or type1 == float:
                return diff
            else:
                s = "Input data must be number or array. (" + '{}'.format(str(type1)) + ")"
                raise TypeError(s)
            

        x0 = [1, 1]

        return minimize(sqr_diff, x0)['x'] # The tuple (gamma_min, nu_min)

    else: 
        raise TypeError("Input data not of the same type")


def bind_array(arr, offset):

    '''
    Vind maximale waarde uit een input array en geef alle waardes terug in de "buurt" van 
    dat maximum. Buurt wordt gedefinieerd door de offset parameter.
    -------------------------------------------------------------------------------------
    '''

    max_value = arr.max()
    max_index = np.where(arr==max_value)[0][0]
    ub = max_index + offset + 1
    lb = max_index - offset

    return arr[lb:ub]




def minimize_chi_wiggle(files, offset):

    '''
    Poging om goede waardes van de critische exponenten te vinden door het verschil tussen
    de "toppen" van elke twee opeenvolgende data sets te minimalizeren. De "toppen" worden 
    gevonden door de bind_array functie. Het werkt alleen niet.
    --------------------------------------------------------------------------------------
    '''

    file1 = files[0]
    file1_chi_array = generate_chi_array(file1)


    gamma_list = []
    nu_list = []

    buff = file1 # buffer
   

    # Note: generate_chi_array returns: [0] = chi_array; [1] = L; [2] = temperatures
    #                                                            [3] = fits


    for i, f in enumerate(files):

        if i is not 1:
            chi_1 = generate_chi_array(f) 
            chi_2 = generate_chi_array(buff)
            buff = f

            chi_1_arr = bind_array(chi_1[0], offset)
            chi_2_arr = bind_array(chi_2[0], offset)

            x_min = min_sqr_diff(chi_1_arr, chi_1[1], chi_2_arr, chi_2[1])

            gamma_list.append(x_min[0])
            nu_list.append(x_min[1])

        else:
            pass

                   

    gamma_mean = np.array(gamma_list).mean()
    nu_mean = np.array(nu_list).mean()
        
    return (gamma_mean, nu_mean)


def plot_chi(c_fits, directory, name, offset, gamma, nu):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for c in c_fits:
#        label_string = str(int(np.sqrt(c[2][0]))) + 'x' + str(int(np.sqrt(c[2][0]))) # Werkt voor 2D
        ax2.plot(c[1], c[0], 'o-') #, label=label_string)

        T.append(list(c[1]))
        C = C + list(c[0])

    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])


    title = "offset = " + '{}'.format(offset) + "; gamma = " + '{}'.format(gamma) + "; nu = " + '{}'.format(nu)
    ax2.set_title(title)
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
    parser.add_argument('-o', '--offset', default = 5,  type=int)


    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()
















    
