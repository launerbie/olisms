#!/bin/usr/env python3

import numpy as np
import random as rnd

def main():

    i = Ising(10,10)
    print(i.calc_energy())
    i.printlattice()



class Ising(object):
    
    def __init__(self, rij=10, kolom=10):
        self.makegrid(rij, kolom)
        self.rij = rij
        self.kolom = kolom
        self.shape = (rij, kolom)

    def makegrid(self, x, y):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        """
        #entries = [rnd.choice([-1,1]) for i in range (x*y)]
        #self.grid = np.array(entries).reshape(x, y)

        self.grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)

    def calc_energy(self, energy=0):
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in collumn with 
        first in collum (torus).
        """
        x = self.rij        #For less writing
        y = self.kolom        
        grd = self.grid

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
        return energy

    def printlattice(self):
        x = self.rij        
        y = self.kolom        
        grd = self.grid

        str_ising = np.empty(grd.size , dtype='str').reshape( grd.shape )
        str_ising[ np.where(grd == 1) ] = '#'
        str_ising[ np.where(grd == -1) ] = ' '
        print(str_ising)


if __name__ == "__main__":
    main()
