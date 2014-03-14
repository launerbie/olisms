#!/bin/usr/env python3
import argparse
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

def main():

    i = Ising(40, 40)
#    print(i.calc_energy())
#    i.printlattice()
    i.evolve(args.iterations, args.beta)
#    i.plotevolution()

class Ising(object):
    
    def __init__(self, rij=10, kolom=10):
        self.makegrid(rij, kolom)
        self.rij = rij
        self.kolom = kolom
        self.shape = (rij, kolom)
        self.total_energy = self.calc_energy()


    def makegrid(self, x, y):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        """
        #entries = [rnd.choice([-1,1]) for i in range (x*y)]
        #self.grid = np.array(entries).reshape(x, y)

        self.grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)

    def calc_energy(self):
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in collumn with 
        first in collum (torus).
        """
        x = self.rij        #For less writing
        y = self.kolom        
        grd = self.grid

        energy = 0
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
                
        return 2*d_energy 
    
    def magnetization(self):
        return self.grid.sum()

    def boltzmann(self, delta_energy, beta=1000):
        return np.exp(beta*delta_energy) 

    
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

        

    def evolve(self, iteraties, beta):
        i = 0
        energy_as_function_of_time = []

        while i < iteraties:
            site = self.choose_site() # choose random site at the beginning of each iteration
            delta_e = self.delta_energy(site) # calculate energy change if that spin were flipped
            probability = self.boltzmann(delta_e, beta) 
            flipped = self.flip(probability, site) # flip spin with probability exp(beta*delta_e)
                                                   # and return boolean indicating if flipped.
            
            
            if flipped and delta_e != 0:
                self.total_energy = self.total_energy + delta_e
                if i % 500:
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

        fig = plt.figure()
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
    parser.add_argument('-b', '--beta', default=1000, type=float, help="The Beta Parameter") 
    parser.add_argument('-i', '--iterations', default=100000, type=int, help="Number of iterations, default: 100000") 
    # Add your arguments here. See below for examples.
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    np.set_printoptions(threshold=np.nan, linewidth= 300)
    main()





































