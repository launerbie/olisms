#!/usr/bin/env python

import h5py
import numpy as np
import matplotlib.pyplot as plt

def main():
 
    f = h5py.File('test_dev_1.hdf5')
    
    autocorrelation(f) 

def variance(input_array):

    arr_squared = input_array**2
    
    var = arr_squared.mean() - (input_array.mean())**2

    return var


def print_progress(counter, total, stepsize):
    
    if (counter)%(total/stepsize) == 0:
#        percent = percent + 100/stepsize
        print('PROGRESS: ', counter/(total/100), '%')

def autocorrelation(f):
    sims = f.values()

    simulation_number = 1
    for s in sims:
        
        print("\n", 'COMPUTING AUTOCORRELATION: SIMULATION', simulation_number,)
        simulation_number = simulation_number + 1

        figname = str(s.name)

        energy = s['energy'].value
        
        '''
        Hier implementeren: c(Dt) = <(E(t+Dt)-<E>)*(E(t)-<E>)>_t
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
 
        plt.savefig('plots2'+figname+"autocorrelation"+".png")   

main()            
