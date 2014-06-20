#!/usr/bin/env python

import os
import h5py
import numpy
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm


def main():
    directory = 'figures/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    f = h5py.File(args.filename) 

    def plot_netmagnitization_vs_iteration():
        fig = plt.figure(figsize=(8,8))        
        ax = fig.add_subplot(111)
    
        for sim in f.values(): #Each sim corresponds to a simulation at some Temperature
            T = sim['temperature'][0]
            M = sim['magnetization'][-1000:]
            #E = sim['energy'].value
            iterations = sim['iterations'][-1000:]
            ax.plot(iterations, abs(M), label=str(T))
    
        ax.set_xlabel('iteration')
        ax.set_ylabel('net magnetization')
        ax.legend(loc='best', prop={'size':8})

        if not args.plot:
            plt.savefig('figures/'+str(args.filename)+"net_M_vs_iters"+".png", bbox_inches='tight')
            fig.clf()
            plt.close()
        else:
            plt.show()

    def plot_energy_vs_iteration():
        fig = plt.figure(figsize=(8,8))        
        ax = fig.add_subplot(111)
    
        for sim in f.values(): #Each sim corresponds to a simulation at some Temperature
            T = sim['temperature'][0]
            M = sim['magnetization'][-1000:]
            E = sim['energy'][-1000:]
            iterations = sim['iterations'][-1000:]
            ax.plot(iterations, E, label=str(T))
    
        ax.set_xlabel('iteration')
        ax.set_ylabel('E')
        ax.legend(loc='best', prop={'size':8})

        if not args.plot:
            plt.savefig('figures/'+str(args.filename)+"E_vs_iters"+".png", bbox_inches='tight')
            fig.clf()
            plt.close()
        else:
            plt.show()

    def plot_chi_vs_temperature():
        fig = plt.figure(figsize=(8,8))        
        ax = fig.add_subplot(111)
        
        temperatures = []
        chi = []
        
        for sim in f.values():
            T = sim['temperature'][0]
            temperatures.append(T)
            net_M = abs(sim['magnetization'][-2000:])
            chi.append(T/1600.0*numpy.var(net_M))
            
        ax.plot(numpy.array(temperatures), chi, marker='o')
        ax.set_xlabel('1/T')
        ax.set_ylabel('chi')

        if not args.plot:
            plt.savefig('figures/'+str(args.filename)+"chi_vs_T"+".png", bbox_inches='tight')
            fig.clf()
            plt.close()
        else:
            plt.show()

    def plot_all_in_one(MCS0=18000):
        fig = plt.figure(figsize=(8,8))        
        ax1 = fig.add_subplot(221) # E vs MCS
        ax2 = fig.add_subplot(222) # |M| vs MCS
        ax3 = fig.add_subplot(223) # <|M|> vs T
        ax4 = fig.add_subplot(224) # chi vs T

        N = 25
        
        temperatures = []
        avg_net_mags = []
        chi = []

        for sim in f.values(): #Each sim corresponds to a simulation at some Temperature
            T = sim['temperature'][0]
            M = sim['magnetization'][-MCS0:]
            net_M = abs(M)
            E = sim['energy'][-MCS0:]
            iterations = sim['iterations'][-MCS0:]

            temperatures.append(T)
            avg_net_mags.append(numpy.mean(net_M))
            chi.append(T/N*numpy.var(net_M))

            ax1.plot(iterations, E)
            ax2.plot(iterations, net_M)

        ax3.plot(temperatures, numpy.array(avg_net_mags)/N, c='k', marker='o')
        ax4.plot(numpy.array(temperatures), chi, c='k', marker='o')

        if not args.plot:
            plt.savefig('figures/'+str(args.filename)+"summary"+".png", bbox_inches='tight')
            fig.clf()
            plt.close()
        else:
            plt.show()


    #plot_netmagnitization_vs_iteration()
    #plot_energy_vs_iteration()
    #plot_chi_vs_temperature()

    plot_all_in_one()

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


