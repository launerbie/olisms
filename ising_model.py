#!/bin/usr/env python3
import subprocess 
import numpy as np
import random as rnd

def main():

    i = Ising(40, 40)
#    print(i.calc_energy())
#    i.printlattice()
    i.evolve(100000)


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
        Randomly dooses site to flip
        """
        site_x = np.random.randint(0,self.rij)
        site_y = np.random.randint(0,self.kolom)
        
        return site_x, site_y
  
    def delta_energy(self, site):
        
        rechts = self.rij - 1
        onder = self.kolom - 1 
        x, y = site
        
        if not (x == 0 or x == rechts or y == 0 or y == onder): # niet rand ==> midden 
            d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[x-1][y] + self.grid[x][y+1] + self.grid[x][y-1])

        else: # rand
            if not ((x == 0 and (y == 0 or y == onder)) or (x == rechts and (y == 0 or y == onder))): # rand & niet hoek
                if x == 0: # linkerrand
                    d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[rechts][y] + self.grid[x][y+1] + self.grid[x][y-1])

                elif x == rechts: # rechterrand
                    d_energy = -self.grid[x][y]*(self.grid[0][y] + self.grid[x-1][y] + self.grid[x][y+1] + self.grid[x][y-1])

                elif y == 0: # boven
                    d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[x-1][y] + self.grid[x][y+1] + self.grid[x][onder])

                else: # onder
                    d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[x-1][y] + self.grid[x][0] + self.grid[x][y-1])

            else: # rand & hoek ==> hoek
                if (x == 0 and y == 0):
                    d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[rechts][y] + self.grid[x][y+1] + self.grid[x][onder])

                elif (x == 0 and y == onder):
                    d_energy = -self.grid[x][y]*(self.grid[x+1][y] + self.grid[rechts][y] + self.grid[x][0] + self.grid[x][y-1])

                elif (x == rechts and y == 0):
                    d_energy = -self.grid[x][y]*(self.grid[0][y] + self.grid[x-1][y] + self.grid[x][y+1] + self.grid[x][onder])

                else:
                    d_energy = -self.grid[x][y]*(self.grid[0][y] + self.grid[x-1][y] + self.grid[x][0] + self.grid[x][y-1])
                
        return 2*d_energy


    def boltzmann(self, delta_energy, beta=0.0001):
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

        

    def evolve(self, iteraties):
        i = 0
        while i < iteraties:
            site = self.choose_site()
            delta_e = self.delta_energy(site)
            probability = self.boltzmann(delta_e)
            flipped = self.flip(probability, site)
            
            
            if flipped and delta_e != 0:
                self.total_energy = self.total_energy + delta_e
                if i % 500:
                    self.printlattice()

            i = i + 1
         

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


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan, linewidth= 120)
    main()



