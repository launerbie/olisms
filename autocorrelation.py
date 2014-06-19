#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
 
    f = h5py.File('test_new_autocorr.hdf5')
    
    autocorrelation_energy(f) 

def variance(input_array):

    arr_squared = input_array**2
    
    var = arr_squared.mean() - (input_array.mean())**2

    return var


def print_progress(counter, total, stepsize):
    
    if (counter)%(total/stepsize) == 0:
        print('PROGRESS: ', counter/(total/100), '%')

def autocorrelation_energy(f):
    sims = f.values()

    simulation_number = 1
    for s in sims:
        
        print("\n", 'COMPUTING AUTOCORRELATION: SIMULATION', simulation_number,)
        simulation_number = simulation_number + 1

        figname = str(s.name)

        energy = s['energy'].value
        
        '''
        Hier implementeren: c_e(Dt) = <(E(t+Dt)-<E>)*(E(t)-<E>)>_t
        '''

        size_energy = s['energy'].shape[0]
        
        avg_energy = s['energy'].value.mean()
 
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
        ax.set_ylabel('correlation')
 
        plt.savefig('plots2'+figname+"energy_autocorrelation"+".png")   

def autocorrelation_magnetization(f):
    sims = f.values()

    simulation_number = 1
    for s in sims:
        
        print("\n", 'SIMULATION', simulation_number,)
        simulation_number = simulation_number + 1

        figname = str(s.name)

        magnetization = s['magnetization'].value
        
        '''
        Hier implementeren: c_m(Dt) = <(M(t+Dt)-<M>)*(M(t)-<M>)>_t
        '''

        size_magnetization = s['magnetization'].shape[0]
        
        avg_magnetization = s['magnetization'].value.mean()
 
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
        ax.set_ylabel('c_e')
 
        plt.savefig('plots2'+figname+"magnetization_autocorrelation"+".png")   

main()            
