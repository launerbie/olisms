#!/usr/bin/env python

import argparse
import numpy as np

def main():
    #TODO: remove main(). 
    i = Ising(args.x, args.y, args.bfield, args.temperature, printit=args.printit)
    i.evolve(args.iterations)

class Ising(object):
    """
    Parameters
    ----------
    rij : number of rows in lattice
    kolom: number of columns in lattice
    b_field: strength of the uniform b-field
    temperature: temperature 
    
    """ 
    def __init__(self, rij=40, kolom=40, b_field=0.0, temperature=0.01, 
                 handler=None, h5path=None, printit=None):
        self.makegrid(rij, kolom)
        self.rij = rij
        self.kolom = kolom
        self.shape = (rij, kolom)
        self.b_field = b_field
        self.temperature = temperature
        self.total_energy = self.calc_energy()
        self.handler = handler 
        self.h5path = h5path 
        self.printit = printit

        if (self.handler and self.h5path) is not None:
            self.handler.append(self.temperature, self.h5path+'temperature')
            self.handler.append(self.b_field, self.h5path+'bfield')

    @property
    def beta(self):
        return 1.0/self.temperature

    def makegrid(self, x, y):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        """
        self.grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)

    def calc_energy(self): 
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in column with 
        first in column (torus).

        """
        x = self.rij        
        y = self.kolom        
        grd = self.grid

        energy = self.b_field * self.magnetization()

        for (i, j), value in np.ndenumerate(grd):
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
        #TODO: 3D array
        site_x = np.random.randint(0,self.rij)
        site_y = np.random.randint(0,self.kolom)
        
        return site_x, site_y
  
    def delta_energy(self, site):
        """
        Berekent verandering in energie als gevolg van het omdraaien van het 
        teken (spin flip) op de positie "site".

        """
        LR = self.rij - 1  # LR: Laatste Rij Index
        LK = self.kolom - 1  # LK: Laatste Kolom Index
        x, y = site
        g = self.grid
        
        if not (x == 0 or x == LR or y == 0 or y == LK): # niet rand ==> midden 
            d_energy = -g[x][y]*(g[x+1][y] + g[x-1][y] + g[x][y+1] + g[x][y-1])

        else: #dan rand
            if not ((x == 0 and (y == 0 or y == LK)) or (x == LR and (y == 0 or y == LK))): #dan niet hoek
                if x == 0: 
                    d_energy = -g[x][y] * (g[x+1][y] + g[LR][y]  + g[x][y+1] + g[x][y-1])

                elif x == LR: 
                    d_energy = -g[x][y] * (g[0][y]   + g[x-1][y] + g[x][y+1] + g[x][y-1])

                elif y == 0: 
                    d_energy = -g[x][y] * (g[x+1][y] + g[x-1][y] + g[x][y+1] + g[x][LK])

                else: 
                    d_energy = -g[x][y] * (g[x+1][y] + g[x-1][y] + g[x][0]   + g[x][y-1])

            else: # dan hoek
                if (x == 0 and y == 0):
                    d_energy = -g[x][y] * (g[x+1][y] + g[LR][y]  + g[x][y+1] + g[x][LK])

                elif (x == 0 and y == LK):
                    d_energy = -g[x][y] * (g[x+1][y] + g[LR][y]  + g[x][0]   + g[x][y-1])

                elif (x == LR and y == 0):
                    d_energy = -g[x][y] * (g[0][y]   + g[x-1][y] + g[x][y+1] + g[x][LK])

                else:
                    d_energy = -g[x][y] * (g[0][y]   + g[x-1][y] + g[x][0]   + g[x][y-1])
                
        return -2*d_energy + 2*self.b_field*g[x][y]  


    def magnetization(self):
        return self.grid.sum()

    def boltzmann(self, delta_energy):
        return np.exp(-self.beta*delta_energy) 

    
    def flip(self, prob, site):
        x, y = site
        determinant = np.random.ranf() #random flt from uniform distr (0,1).
    
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

                if self.handler is not None and self.h5path is not None:
                    self.handler.append(np.array(site), self.h5path+'sites')
                    self.handler.append(i, self.h5path+'iterations')
                    self.handler.append(self.total_energy, self.h5path+'energy')
                    self.handler.append(self.magnetization(), self.h5path+'magnetization')

                if self.printit is not None:
                    if i % self.printit == 0 :
                        self.printlattice()
                
            i = i + 1


    def printlattice(self):
        """
        Prints the lattice.

        """
        grd = self.grid

        str_ising = np.empty(grd.size, dtype='int8').reshape( grd.shape )
        str_ising[ np.where(grd == 1) ] = 1 
        str_ising[ np.where(grd == -1) ] = 8 
        print(str_ising)
        print("\n")

def get_arguments():
    """
    To add arguments, call: parser.add_argument
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', '--temperature', default=0.001, type=float,
                        help="The Temperature") 
    parser.add_argument('-i', '--iterations', default=100000, type=int,
                        help="Number of iterations, default: 100000") 
    parser.add_argument('-b', '--bfield', default=0.00, type=float,
                        help="Uniform external magnetic field, default: 0") 
    parser.add_argument('-y', default=40,type=int,help="number of columns") 
    parser.add_argument('-x', default=40,type=int, help="number of rows") 
    parser.add_argument('-f', '--filename', default='test.hdf5', 
                        help="hdf5 output file name") 
    parser.add_argument('-p', '--printit', default=0,type=int,
                        help="print lattice every p flips") 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    #TODO: get maximum linewidth available in terminal and set this in np.set_printoptions
    np.set_printoptions(threshold=np.nan, linewidth= 300)
    print(args)
    main()

