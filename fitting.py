#! /usr/bin/env python

import os
import h5py
import numpy
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.misc import factorial



'''
Een programma dat uit een x aantal input .hdf5 bestanden vier plotjes maakt: M(T), Chi(T) en de
fits van M(T) en Chi(T). In elk plotje zijn x curves, elk corresponderend met een van de .hdf5 
bestanden. Ik heb het gemaakt met het oog op modulariteit ipv snelheid zodat we makkelijk nieuwe
fits en plots daarvan kunnen toevoegen, bijvoorbeeld autocorrelatie en data-collapse. 
------------------------------------------------------------------------------------------------
'''


def main():
  
#    Helaas is de input van de .hdf5 bestanden gehardcode... dat komt omdat ik niet weet hoe
#    ik via de parser de gridsize samen met de corresponderende .hdf5 bestand moet meegeven.


    file1 = (h5py.File('/data/cursus3/wolff_data/wolff_20x20_10kiterations_50steps_2.hdf5'), 400)
    file2 = (h5py.File('/data/cursus3/wolff_data/wolff_30x30_10kiterations_50steps.hdf5'), 900)
    file3 = (h5py.File('/data/cursus3/wolff_data/wolff_40x40_10kiterations_50steps.hdf5'), 1600)
    file4 = (h5py.File('/data/cursus3/wolff_data/wolff_50x50_10kiterations_50steps.hdf5'), 2500)
 
    directory = 'fits'
    if not os.path.exists(directory):
        os.makedirs(directory)

    name = args.plotname  # Verplicht argument
 
    Files = [file1, file2, file3, file4]

    mags_fits = []
    chi_fits = []

    for f in Files:
        mags_fits.append(fit_mags(f[0], f[1]))
        chi_fits.append(fit_chi(f[0], f[1]))

    plot_mags(mags_fits, directory, name)
    plot_chi(chi_fits, directory, name)

    plot_mags_fits(mags_fits, directory, name)
    plot_chi_fits(chi_fits, directory, name)




def fit_arctan(x_range, data):

    '''
    De vorm van de magnetizatie data is grofweg een arctan(-x)
    ----------------------------------------------------------
    '''

    def func(x, a, b, c, d):
        return a * numpy.arctan(-b * (x - c)) + d

    popt, pcov = curve_fit(func, x_range, data)

    return func(x_range, popt[0], popt[1], popt[2], popt[3])




def fit_gauss(x_range, data):

    '''
    Voor de Chi-data is een gaussiaan meest voor de hand liggend.
    -------------------------------------------------------------
    '''

    def func(x, a, b, c, d):
        return a * numpy.exp(-((x - b)**2)/(2 * c**2)) + d

    popt, pcov = curve_fit(func, x_range, data)

    return func(x_range, popt[0], popt[1], popt[2], popt[3])




def fit_mags(hdf5_file, number_of_spins):

    temperatures = []
    avg_net_mags = []
    list_of_gridsize = []   

    for sim in hdf5_file.values():
        if (number_of_spins == None):
            N = sim['lattice_size'][0]
        else:
            N = number_of_spins 
        MCS0 = 500
        T = sim['temperature'][0]
        M = sim['magnetization'][-MCS0:]
        net_M = abs(M)
        
        temperatures.append(T)
        avg_net_mags.append(numpy.mean(net_M))        
        list_of_gridsize.append(N)


    fit = fit_arctan(numpy.array(temperatures), numpy.array(avg_net_mags)/N)
    return (numpy.array(temperatures), fit, list_of_gridsize, numpy.array(avg_net_mags)/N)




def fit_chi(hdf5_file, number_of_spins):

    temperatures = []
    chi = []
    list_of_gridsize = []
   
    for sim in hdf5_file.values():
        if (number_of_spins == None):
            N = sim['lattice_size'][0]
        else:
            N = number_of_spins 
        MCS0 = 500
        T = sim['temperature'][0]
        M = sim['magnetization'][-MCS0:]
        net_M = abs(M)
        
        temperatures.append(T)
        chi.append(1/(T*N)*numpy.var(net_M))
        list_of_gridsize.append(N)


    fit = fit_gauss(numpy.array(temperatures), chi)
    return (numpy.array(temperatures), fit, list_of_gridsize, chi)




def plot_mags(m_fits, directory, name):

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111)

    T = []  # Voor het bepalen van x_max en x_min om de plot mooi te maken
    M = []  # Voor y_max en y_min. Zie beneden

    for m in m_fits:
        label_string = str(int(numpy.sqrt(m[2][0]))) + 'x' + str(int(numpy.sqrt(m[2][0])))
        ax1.plot(m[0], m[3], 'o-', label=label_string)

        T.append(list(m[0])) # Maak een lijst van lijsten van temperaturen...
        M = M + list(m[3]) # Maak een lijst van alle magnetizaties die voorkomen

    plt.axis([min(max(T)), max(max(T)), min(M), max(M)])           # ...Hier pakken we
        # Ik heb geen bescrijvende naam gekozen voor T en M        # de maxima en minima
        # zodat deze regel niet te groot zou worden.               # van de lijsten om 
                                                                   # een mooie plot te
                                                                   # krijgen
    ax1.set_xlabel('Temperatures')
    ax1.set_ylabel('Net magnetization')
    ax1.legend(loc='best', prop={'size':12})

    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')

    plt.savefig(directory+"/"+str(name)+"_mags"+".png", bbox_inches='tight')




def plot_mags_fits(m_fits, directory, name):

    fig1 = plt.figure(figsize=(8,8))
    ax1 = fig1.add_subplot(111)

    T = []  
    M = []  

    for m in m_fits:
        label_string = str(int(numpy.sqrt(m[2][0]))) + 'x' + str(int(numpy.sqrt(m[2][0])))
        ax1.plot(m[0], m[1], label=label_string)

        T.append(list(m[0])) 
        M = M + list(m[1])

    plt.axis([min(max(T)), max(max(T)), min(M), max(M)]) 
                                                                   
    ax1.set_xlabel('Temperatures')
    ax1.set_ylabel('Net magnetization')
    ax1.legend(loc='best', prop={'size':12})

    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    plt.savefig(directory+"/"+str(name)+"_mags_fits"+".png", bbox_inches='tight')




def plot_chi(c_fits, directory, name):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for c in c_fits:
        label_string = str(int(numpy.sqrt(c[2][0]))) + 'x' + str(int(numpy.sqrt(c[2][0])))
        ax2.plot(c[0], c[3], 'o-', label=label_string)

        T.append(list(c[0]))
        C = C + list(c[3])

    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])
    
   
    ax2.set_xlabel('Temperatures')
    ax2.set_ylabel('Magnetic susceptibility')
    ax2.legend(loc='best', prop={'size':12})
    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    plt.savefig(directory+"/"+str(name)+"_chi"+".png", bbox_inches='tight')




def plot_chi_fits(c_fits, directory, name):

    fig2 = plt.figure(figsize=(8,8))
    ax2 = fig2.add_subplot(111)

    T = []
    C = []

    for c in c_fits:
        label_string = str(int(numpy.sqrt(c[2][0]))) + 'x' + str(int(numpy.sqrt(c[2][0])))
        ax2.plot(c[0], c[1], label=label_string)

        T.append(list(c[0]))
        C = C + list(c[1])

    plt.axis([min(max(T)), max(max(T)), min(C), max(C)])

    ax2.set_xlabel('Temperatures')
    ax2.set_ylabel('Magnetic susceptibility')
    ax2.legend(loc='best', prop={'size':12})

    ax2.xaxis.set_ticks_position('none')
    ax2.yaxis.set_ticks_position('none')
    plt.savefig(directory+"/"+str(name)+"_chi_fits"+".png", bbox_inches='tight')




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('plotname')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()












