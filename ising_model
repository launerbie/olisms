#!/bin/usr/env python3

import numpy as np
import random as rnd

def main():

    i = Ising(3,3)
#    print(i.grid)
#    print(i.calc_energy())
    print(i)



class Ising:
    
    def __init__(self, rij=4, kolom=4):
        self.makegrid(rij, kolom)
        self.rij = rij
        self.kolom = kolom
        self.shape = (rij, kolom)

    #Funtion that makes a numpy array with x rows and y collumns and fills the entries
    #randomly with '1' or '-1'
    def makegrid(self, x, y):
        
        entries = [rnd.choice([-1,1]) for i in range (x*y)]
        self.grid = np.array(entries).reshape(x, y)

    #Function that iterates through the ising array and calculates product of its value 
    #with right and lower neighbor. Boundary conditions link last entry in row with first 
    #in row and last entry in collumn with first in collum (torus).
    def calc_energy(self, energy=0):
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

    def __str__(self):
      
        grd = self.grid
        shape = self.shape
         
        toprint = []
          
        for i in range(grd.shape[0]):
            for j in range(grd.shape[1]):
                if i != self.rij - 1:
                    if grd[i][j] == 1:
                        toprint.append('+')

                    else: 
                        toprint.append('-')
                else:
                    if grd[i][j] == 1:
                        toprint.append('+')

                    else: 
                        toprint.append('-')

                    toprint.append('\n')  
        return str(toprint)
main()
