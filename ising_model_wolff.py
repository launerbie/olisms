#!/usr/bin/env python

import random
import argparse
import numpy as np

def main():
    #TODO: remove main(). 
    i = Ising(args.x, args.y, args.bfield, args.temperature, printit=args.printit)
    i.ewolve(args.iterations)

class Ising(object):
    """
    Parameters
    ----------
    rij : number of rows in lattice
    kolom: number of columns in lattice
    b_field: strength of the uniform b-field
    temperature: temperature 
    
    """ 
    def __init__(self, rij=40, kolom=40, b_field=0.0, temperature=10, 
                 handler=None, h5path=None, printit=None, initgrid=None):

        if initgrid is not None:
            self.grid = initgrid
        else:
            self.grid = self.makegrid(rij, kolom)

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
            self.handler.append(np.array(self.grid, dtype='int8'), self.h5path+'initgrid', dtype='int8')

    @property
    def bond_probability(self):
        J = 1
        kB =1
        return 1 - np.exp(-2*J/(kB*self.temperature))

    @property
    def beta(self):
        #kB = 1.3806488e-23 J K^-1
        kB = 1
        return 1.0/(kB*self.temperature)

    def magnetization(self):
        return self.grid.sum()

    def makegrid(self, x, y):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        """
        grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)
        return np.array(grid, dtype='int8')
        

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
        Randomly chooses a site in the grid.
        """
        #TODO: 3D array
        site_x = np.random.randint(0,self.rij)
        site_y = np.random.randint(0,self.kolom)
        
        return site_x, site_y
        

    def ewolve(self, iterations):
        """
        Evolve it using Wolff's algorithm.
        """
        g = self.grid

        i=0
        while i < iterations:
            self.total_energy = self.calc_energy() 

            if args.verbose is True:
                self.printlattice()
                print("Clusters flipped : {}".format(i))
                print("Temperature      : {}".format(self.temperature))
                print("Bond Probability : {}".format(self.bond_probability))

            cluster = list() 
            perimeter_spins = list() 

            seed = self.choose_site() 
            seed_spin = g[seed]
            cluster.append(seed)

            nbrs = self.neighbors(seed)

            for nbr in nbrs:
                if seed_spin == g[nbr]:
                    perimeter_spins.append(nbr)

            while len(perimeter_spins) != 0:

                site_to_test = perimeter_spins[0]
                perimeter_spins.pop(0)

                if seed_spin == g[site_to_test]:
                    determinant = np.random.ranf()

                    if determinant <= self.bond_probability: #then add to cluster
                        cluster.append(site_to_test)

                        nbrs = self.neighbors(site_to_test)
                        for nbr in nbrs:
                            if nbr not in cluster and nbr not in perimeter_spins:
                                perimeter_spins.append(nbr)

            #flip cluster
            g[np.array(cluster)[:,0], np.array(cluster)[:,1]] = -seed_spin

            i += 1

    def neighbors(self, site):
        """
        TODO: maybe this function can be used in delta_energy in the metropolis algorithm?
        """
        LR = self.rij - 1  
        LK = self.kolom - 1
        x, y = site
        if not (x == 0 or x == LR or y == 0 or y == LK): # dan niet rand  
            nbrs = (x+1, y), (x-1, y), (x, y+1), (x, y-1)
        else: #dan rand
            if not ((x == 0 and (y == 0 or y == LK)) or (x == LR and (y == 0 or y == LK))): #dan niet hoek
                if x == 0: 
                    nbrs = (x+1, y), (LR, y), (x, y+1), (x, y-1)
                elif x == LR: 
                    nbrs = (0, y), (x-1, y), (x, y+1), (x, y-1)
                elif y == 0: 
                    nbrs = (x+1, y), (x-1, y), (x, y+1), (x, LK)
                else: 
                    nbrs = (x+1, y), (x-1, y), (x, 0), (x, y-1)
            else: # dan hoek
                if (x == 0 and y == 0):
                    nbrs = (x+1, y), (LR, y), (x, y+1), (x, LK)
                elif (x == 0 and y == LK):
                    nbrs = (x+1, y), (LR, y), (x, 0), (x, y-1)
                elif (x == LR and y == 0):
                    nbrs = (0, y), (x-1, y), (x, y+1), (x, LK)
                else:
                    nbrs = (0, y), (x-1, y), (x, 0), (x, y-1)
        return nbrs

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
    parser.add_argument('-v', '--verbose', action='store_true') 
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

