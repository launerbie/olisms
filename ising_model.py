#!/bin/usr/env python

import argparse
import numpy as np
import random as rnd
from hdf5utils import HDF5Handler
import matplotlib.pyplot as plt

def main():

    i = Ising(args.x, args.y, args.bfield, args.temperature)
#    print(i.calc_energy())
#    i.printlattice()
    i.evolve(args.iterations)
    i.plotevolution()

class Ising(object):
    
    def __init__(self, rij=40, kolom=40, field=0, temperature=0.01):
        self.makegrid(rij, kolom)
        self.rij = rij
        self.kolom = kolom
        self.b_field = field
        self.temperature = temperature
        self.shape = (rij, kolom)
        self.total_energy = self.calc_energy()

    @property
    def beta(self):
        return 1/self.temperature

    def makegrid(self, x, y):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        """
        #entries = [rnd.choice([-1,1]) for i in range (x*y)]
        #self.grid = np.array(entries).reshape(x, y)

        self.grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)

    def calc_energy(self): #b_field parameter toegevoegd op 14/03/2014.
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in collumn with 
        first in collum (torus).
        """
        x = self.rij        #For less writing
        y = self.kolom        
        grd = self.grid
        magnetization = self.magnetization()

        energy = self.b_field * magnetization
        for i in range(grd.shape[0]):
            for j in range(grd.shape[1]):
            
                if i == (x - 1) and j == (y - 1):           
                    energy = energy + grd[i][j]*grd[0][j] + grd[i][j]*grd[i][0]
            
                elif i == (x - 1):
                    energy = energy + grd[i][j]*grd[0][j] + grd[i][j]*grd[i][j+1]  
            
                elif j == y - 1:
                    energy = energy + grd[i][j]*grd[i+1][j] + grd[i][j]*grd[i][0]  

                else:
                    energy = energy + grd[i][j]*grd[i+1][j] + grd[i][j]*grd[i][j+1] 
        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.


    def choose_site(self):
        """
        Randomly chooses site to flip
        """
        site_x = np.random.randint(0,self.rij)
        site_y = np.random.randint(0,self.kolom)
        
        return site_x, site_y
  
    def delta_energy(self, site):
       
        """
        Berekent verandering in energie als gevolg van het omdraaien van het teken (spin flip)
        op de positie aangegeven door het verplichte argument "site".
        """
 
        rechts = self.rij - 1  # zou eigenlijk onder moeten heten, heeft geen invloed op programma.
        onder = self.kolom - 1  # zou eigenlijk rechts moeten heten.
        x, y = site
        grd = self.grid
        
        if not (x == 0 or x == rechts or y == 0 or y == onder): # niet rand ==> midden 
            d_energy = -grd[x][y]*(grd[x+1][y] + grd[x-1][y] + grd[x][y+1] + grd[x][y-1])

        else: # rand
            if not ((x == 0 and (y == 0 or y == onder)) or (x == rechts and (y == 0 or y == onder))): # rand & niet hoek
                if x == 0: # linkerrand
                    d_energy = -grd[x][y]*(grd[x+1][y] + grd[rechts][y] + grd[x][y+1] + grd[x][y-1])

                elif x == rechts: # rechterrand
                    d_energy = -grd[x][y]*(grd[0][y] + grd[x-1][y] + grd[x][y+1] + grd[x][y-1])

                elif y == 0: # boven
                    d_energy = -grd[x][y]*(grd[x+1][y] + grd[x-1][y] + grd[x][y+1] + grd[x][onder])

                else: # onder
                    d_energy = -grd[x][y]*(grd[x+1][y] + grd[x-1][y] + grd[x][0] + grd[x][y-1])

            else: # rand & hoek ==> hoek
                if (x == 0 and y == 0):
                    d_energy = -grd[x][y]*(grd[x+1][y] + grd[rechts][y] + grd[x][y+1] + grd[x][onder])

                elif (x == 0 and y == onder):
                    d_energy = -grd[x][y]*(grd[x+1][y] + grd[rechts][y] + grd[x][0] + grd[x][y-1])

                elif (x == rechts and y == 0):
                    d_energy = -grd[x][y]*(grd[0][y] + grd[x-1][y] + grd[x][y+1] + grd[x][onder])

                else:
                    d_energy = -grd[x][y]*(grd[0][y] + grd[x-1][y] + grd[x][0] + grd[x][y-1])
                
        return -2*d_energy + 2*self.b_field*grd[x][y] # toegevoegd: -2B*site; verandering in energie als 
                                                     # gevolg van spin flip in extern veld van sterkte B

    def magnetization(self):
        return self.grid.sum()

    def boltzmann(self, delta_energy):
        return np.exp(-self.beta*delta_energy) 

    
    def flip(self, prob, site):
        
        x, y = site
        determinant = np.random.ranf() # Retruns uniform random float from range (0,1).
    
        if prob >= 1:
            self.grid[x][y] = -self.grid[x][y]
            return True

        else:
            if determinant <= prob:                     
                self.grid[x][y] = -self.grid[x][y]
                return True
            else:
                return False

        

    def evolve(self, iteraties):

        with HDF5Handler(args.filename) as h:

            energy_as_function_of_time = []
            i = 0
            while i < iteraties:

                if i % 1000 == 0:
                    print(i)
 
                site = self.choose_site() # choose random site at the beginning of each iteration
                delta_e = self.delta_energy(site) # calculate energy change if that spin were flipped
                probability = self.boltzmann(delta_e) 
                flipped = self.flip(probability, site) # flip spin with probability exp(beta*delta_e)
                                                       # and return boolean indicating if flipped.
                
                
                if flipped and delta_e != 0:
                    self.total_energy = self.total_energy + delta_e

                    h.append(np.array(site), 'site')
                    h.append(np.array(i), 'iteration')
                    h.append(np.array(self.total_energy), 'energy')

                    if args.printit != 0:
                        if i % args.printit == 0 :
                            self.printlattice()
                    
                energy_as_function_of_time.append(self.total_energy) # For plotting E(t). Builds list of total energy per iteration.

                i = i + 1
             
        self.time_variable = np.arange(iteraties) # For plotting E(t). This array will be the time axis
        self.energy_variable = np.array(energy_as_function_of_time) # For plotting E(t). This array will be the energy axis

    def plotevolution(self):
        """
        Makes a plot of total system energy as a funtion of time (iterations) based on the data collected in 
        the function "evolve". The plot is saved under the name "evolution.png"
        """

        E = self.energy_variable
        t = self.time_variable

        fig = plt.figure(figsize=(20,4))
        ax = fig.add_subplot(111)

        ax.plot(t, E)
        plt.savefig("evolution")



    def printlattice(self):
        """
        Prints the lattice.
        """
       
        x = self.rij        
        y = self.kolom        
        grd = self.grid

        str_ising = np.empty(grd.size, dtype='int').reshape( grd.shape )
        #str_ising = np.empty(grd.size , dtype='str').reshape( grd.shape )
        str_ising[ np.where(grd == 1) ] = 1 
        str_ising[ np.where(grd == -1) ] = 8 
        print(str_ising)
        print("\n")

def get_arguments():
    """
    To add arguments, call: parser.add_argument
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--temperature', default=0.001, type=float, help="The Temperature") 
    parser.add_argument('-i', '--iterations', default=100000, type=int, help="Number of iterations, default: 100000") 
    parser.add_argument('-b', '--bfield', default=0.00, type=float, help="Uniform external magnetic field, default: 0") 
    parser.add_argument('-y', default=40,type=int, help="number of collumns (width)") 
    parser.add_argument('-x', default=40,type=int, help="number of rows (height)") 
    parser.add_argument('-f', '--filename', default='test.hdf5', help="hdf5 output file name") 
    parser.add_argument('-p', '--printit', default=0,type=int, help="print lattice every p flips") 
    # Add your arguments here. See below for examples.
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    np.set_printoptions(threshold=np.nan, linewidth= 300)
    print(args)
    main()















