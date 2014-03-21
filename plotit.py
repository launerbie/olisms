#!/usr/bin/env python

import os
import h5py
import numpy
import argparse
import matplotlib.pyplot as plt

def main():
    f = h5py.File(args.filename) 

    plotdata(f)


def plotdata(f):
    sims = f.values()

    for s in sims:
        figname = str(s.name)

        temperature = s['temperature'][0]
        bfield = s['bfield'][0]
 
        energy = s['energy'].value
        magnetization = s['magnetization'].value
        iterations = s['iterations'].value

        fig = plt.figure(figsize=(8,16))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(iterations, energy)
        ax2.plot(iterations, magnetization)

        ax1.set_xlabel('iteration')
        ax1.set_ylabel('energy')
        #ax1.legend(loc='best', prop={'size':6})
        ax1.set_title('T={}  B={}'.format(str(temperature), str(bfield)))

        ax2.set_xlabel('iteration')
        ax2.set_ylabel('magnetization')
        #ax2.legend(loc='best', prop={'size':6})

        fig.subplots_adjust(wspace=0.1, hspace=0.1)

        plt.savefig('plots/'+figname+".png", bbox_inches='tight')
        fig.clf()
        plt.close()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename', metavar="HDF5 FILENAME")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    print(args)
    main()


