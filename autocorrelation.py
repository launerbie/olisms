#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
 
    f = h5py.File(args.filename)
    
#    autocorrelation_energy(f, args.n) 
    autocorrelation_magnetization(f, args.n) 
    

def variance(input_array):

    arr_squared = input_array**2
    
    var = arr_squared.mean() - (input_array.mean())**2

    return var


def print_progress(counter, total, stepsize):
    
    if (counter)%(total/stepsize) == 0:
        print('PROGRESS: ', counter/(total/100), '%')

def autocorrelation_energy(f, number_of_spins):
    sims = f.values()

    simulation_number = 1
    for s in sims:
        
        print("\n", 'COMPUTING AUTOCORRELATION: SIMULATION', simulation_number,)
        simulation_number = simulation_number + 1

        figname = str(s.name)

        if (number_of_spins == None):
            N = s['lattice_size'][0]
        else:
            N = number_of_spins #Included so that we can plot "old" data sets as well
      
        MSC0 = int(10*N)

        energy = s['energy'][-MSC0:]
        
        '''
        Hier implementeren: c_e(Dt) = <(E(t+Dt)-<E>)*(E(t)-<E>)>_t
        '''

        size_energy = len(s['energy'][-MSC0:])
        
        avg_energy = s['energy'][-MSC0:].mean()
 
        delta_t = np.arange(size_energy)
        correlation = [] 
        
        var = variance(energy)

        for k in delta_t:
            summ = 0
            i = 0

            print_progress(k, size_energy, 10)

            while i+k < size_energy:
                summ = summ + (energy[i + k] - avg_energy)*(energy[i] - avg_energy)
                i = i + 1
            
            c_van_delta_t = summ/var 
            correlation.append(c_van_delta_t)

        corr_array = np.array(correlation)
        

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(delta_t, corr_array)

        ax.set_xlabel('delta_t')
        ax.set_ylabel('c_e')
 
        plt.savefig('plots2'+figname+"energy_autocorrelation"+".png")   

def autocorrelation_magnetization(f, number_of_spins):
    sims = f.values()

    simulation_number = 1
    for s in sims:
        
        print("\n", 'SIMULATION', simulation_number,)
        simulation_number = simulation_number + 1

        figname = str(s.name)

        if (number_of_spins == None):
            N = s['lattice_size'][0]
        else:
            N = number_of_spins #Included so that we can plot "old" data sets as well
       
        MSC0 = int(10*N)

        magnetization = s['magnetization'][-MSC0:]
        
        '''
        Hier implementeren: c_m(Dt) = <(M(t+Dt)-<M>)*(M(t)-<M>)>_t
        '''

        size_magnetization = len(s['magnetization'][-MSC0:])
        
        avg_magnetization = s['magnetization'][-MSC0:].mean()
 
        delta_t = np.arange(size_magnetization)
        correlation = [] 
        
        var = variance(magnetization)

        for k in delta_t:
            summ = 0
            i = 0

            print_progress(k, size_magnetization, 10)

            while i+k < size_magnetization:
                summ = summ + (magnetization[i + k] - avg_magnetization)*(magnetization[i] - avg_magnetization)
                i = i + 1
            
            c_van_delta_t = summ/var 
            correlation.append(c_van_delta_t)

        corr_array = np.array(correlation)
        

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(delta_t, corr_array)

        ax.set_xlabel('delta_t')
        ax.set_ylabel('c_m')
 
        plt.savefig('plots3'+figname+"magnetization_autocorrelation"+".png")   

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar="HDF5 FILENAME")
    parser.add_argument('-n', default = None, type = int,
                       help="Number of spins. Specify only if hdf5 file does not include this info.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()



