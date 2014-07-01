#!/usr/bin/env python

import shutil
import argparse
import numpy as np

def main():
    #TODO: remove main(). 
    if args.full:
        args.x = screensize.lines - 8 
        args.y = (screensize.columns//2) -2
    
    i = Ising(args.x, args.y, args.bfield, args.temperature, 
              printit=args.printit, mode=args.mode, aligned=args.aligned)

    i.evolve(args.iterations)

class Ising(object):
    def __init__(self, rij=40, kolom=40, b_field=0.0, temperature=10, 
                 handler=None, h5path=None, printit=None, aligned=False,
                 mode='metropolis'):
        """ 
        Parameters
        ----------
        rij : number of rows in lattice
        kolom: number of columns in lattice
        b_field: strength of the uniform b-field
        temperature: temperature 
        handler: HDF5Handler instance
        h5path: unix-style path, used as address in the hdf5 file
        initgrid: set an initial ndarray as grid
        mode : algorithm to use. Choices are: ['metropolis', 'wolff']
    
        """
        if mode == 'metropolis':
            self.evolve = self.evolve_metropolis
        elif mode == 'wolff':
            self.evolve = self.evolve_wolff
        else:
            raise ValueError("Unknown mode")

        self.grid = self.makegrid(rij, kolom, aligned=aligned)

        self.rij = rij
        self.kolom = kolom
        self.shape = (rij, kolom)
        self.b_field = b_field
        self.temperature = temperature
        self.total_energy = self.calc_energy()
        self.handler = handler 
        self.h5path = h5path 
        self.printit = printit
        self.ptable = self.make_probability_table()
        self.lattice_size = rij*kolom

        if (self.handler and self.h5path) is not None:
            self.handler.append(self.temperature, self.h5path+'temperature')
            self.handler.append(self.lattice_size, self.h5path+'lattice_size')
            self.handler.append(np.array(self.grid, dtype='int8'), 
                                self.h5path+'initgrid', dtype='int8')

    def make_probability_table(self):
        delta_energies = [-8, -4, 0, 4, 8] #TODO: un-hardcode
        ptable = dict() 
        for dE in delta_energies:
            ptable.update({dE:np.exp(-dE/self.temperature)}) 
        return ptable


    #deprecated
    #@property
    #def beta(self):
    #    #kB = 1.3806488e-23 J K^-1
    #    kB = 1
    #    return 1.0/(kB*self.temperature)

    #deprecated
    #@property
    #def bond_probability(self):
    #    J = 1
    #    kB =1
    #    return 1 - np.exp(-2*J/(kB*self.temperature))

    #deprecated
    #def boltzmann(self, delta_energy):
    #    return np.exp(-self.beta*delta_energy) 

    @property
    def magnetization(self):
        return self.grid.sum()

    def choose_site(self):
        """
        Randomly chooses site to flip
        """
        #TODO: 3D array
        site_x = np.random.randint(0,self.rij)
        site_y = np.random.randint(0,self.kolom)
        
        return site_x, site_y

    def makegrid(self, x, y, aligned=False):
        """ 
        Function that makes a numpy array with x rows and y columns and fills 
        the entries randomly with '1' or '-1'
        If aligned = True, a grid of ones is returned.
        """
        if aligned:
            grid = np.ones((x, y))
            return np.array(grid, dtype='int8')
        else:
            grid = np.random.choice([-1, 1], size=x*y).reshape(x, y)
            return np.array(grid, dtype='int8')
        
    def neighbors(self, site, boundary='periodic'):
        """
        TODO: return neighbors based on boundary argument.
      
        Returns a list of sites which are the neighbors of 'site' based on
        the boundary condition.

        Return
        ------
        tuple: (below nbr), (above nbr), (right nbr), (left nbr)
        
        Here: "below", "above", "right" and "left" mean:

                   j-1         j         j+1

                ---------------------------------
        i-1     |        |    above   |         |
                |        |            |         |
                ---------------------------------
         i      |  left  |    site    |  right  |
                |        |   (i, j)   |         |
                ---------------------------------
        i+1     |        |            |         |
                |        |    below   |         |
                ---------------------------------


        So if you consider the ndarray a=numpy.arange(9).reshape(3,3):

                >>> a
                array([[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]])
                >>> print(a[0,0], a[1,1], a[2,2])
                0 4 8

        Then the left neighbor of 4 is 3.

        """
        LR = self.rij - 1
        LK = self.kolom - 1
        i, j = site
        if not (i == 0 or i == LR or j == 0 or j == LK): # dan niet rand  
            nbrs = (i+1, j), (i-1, j), (i, j+1), (i, j-1)
        else: #dan rand
            if not ((i == 0 and (j == 0 or j == LK)) or\
                    (i == LR and (j == 0 or j == LK))): #dan niet hoek
                if i == 0:
                    nbrs = (i+1, j), (LR, j), (i, j+1), (i, j-1)
                elif i == LR:
                    nbrs = (0, j), (i-1, j), (i, j+1), (i, j-1)
                elif j == 0:
                    nbrs = (i+1, j), (i-1, j), (i, j+1), (i, LK)
                else:
                    nbrs = (i+1, j), (i-1, j), (i, 0), (i, j-1)
            else: # dan hoek
                if (i == 0 and j == 0):
                    nbrs = (i+1, j), (LR, j), (i, j+1), (i, LK)
                elif (i == 0 and j == LK):
                    nbrs = (i+1, j), (LR, j), (i, 0), (i, j-1)
                elif (i == LR and j == 0):
                    nbrs = (0, j), (i-1, j), (i, j+1), (i, LK)
                else:
                    nbrs = (0, j), (i-1, j), (i, 0), (i, j-1)
        return nbrs


    def calc_energy(self): 
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in column with 
        first in column (torus).

        """
        g = self.grid
        energy = self.b_field * self.magnetization

        for site, value in np.ndenumerate(g):
            below, above, right, left = self.neighbors(site)
            energy = energy + g[site]*( g[right] + g[below] )

        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.


    def delta_energy(self, site):
        """
        Berekent verandering in energie als gevolg van het omdraaien van het 
        teken (spin flip) op de positie "site".

        """
        g = self.grid
        below, above, right, left = self.neighbors(site)
        d_energy = -2*(-g[site] * (g[below] + g[above] +g[right] +g[left]))
        return d_energy  

    
    def flip(self, prob, site):
        """ Flip 'site' with probability 'prob'."""
        determinant = np.random.ranf() #random flt from uniform distr (0,1).

        if prob >= 1:
            self.grid[site] = -self.grid[site]
            return True

        else:
            if determinant <= prob:                     
                self.grid[site] = -self.grid[site]
                return True
            else:
                return False

        
    def evolve_metropolis(self, iterations):
        """
        Evolve it using Metropolis.
        """
        i = 0
        flipcount = 0
         
        while i < iterations:
            site = self.choose_site() 
            delta_e = self.delta_energy(site) 
            probability = self.ptable[delta_e]
            #probability = self.boltzmann(delta_e) #deprecated
            flipped = self.flip(probability, site) 
            
            if flipped and delta_e != 0:
                flipcount += 1 
                self.total_energy = self.total_energy + delta_e

            if self.printit is not None:
                if i % self.printit == 0 :
                    self.printlattice()
                    print("Temperature    : {}".format(self.temperature))
                    print("B-Field        : {}".format(self.b_field))
                    print("Iterations     : {}".format(i))
                    print("Energy         : {}".format(self.total_energy))
                    print("Magnetization  : {}".format(self.magnetization))
                    print("flips/iters    : {}/{}".format(flipcount, i))
                    print("x/y    : {}/{}".format(self.rij, self.kolom))

            if self.handler is not None and self.h5path is not None:
                self.handler.append(np.array(site), self.h5path+'sites', dtype='int16')
                self.handler.append(i, self.h5path+'iterations', dtype='int64')
                self.handler.append(self.total_energy, self.h5path+'energy')
                self.handler.append(self.magnetization, self.h5path+'magnetization')

            i = i + 1


    def evolve_wolff(self, iterations):
        """
        Ewolve it using Wolff's algorithm.
        """
        g = self.grid

        J = 1
        kB = 1
        bond_probability = 1 - np.exp(-2*J/(kB*self.temperature))

        i=0
        while i < iterations:
            self.total_energy = self.calc_energy()

            cluster = list()
            perimeter = list()

            seed = self.choose_site()
            seed_spin = g[seed]
            cluster.append(seed)

            nbrs = self.neighbors(seed)

            for nbr in nbrs:
                if seed_spin == g[nbr]:
                    perimeter.append(nbr)

            while len(perimeter) != 0:

                site_to_test = perimeter[0]
                perimeter.pop(0)

                if seed_spin == g[site_to_test]:
                    determinant = np.random.ranf()

                    if determinant <= bond_probability: 
                        cluster.append(site_to_test)

                        nbrs = self.neighbors(site_to_test)
                        for nbr in nbrs:
                            if nbr not in cluster and nbr not in perimeter:
                                perimeter.append(nbr)

            #flip cluster
            g[np.array(cluster)[:,0], np.array(cluster)[:,1]] = -seed_spin

            if self.printit is not None:
                if i % self.printit == 0 :
                    self.printlattice()
                    print("Temperature      : {}".format(self.temperature))
                    print("B-Field          : {}".format(self.b_field))
                    #print("Bond Probability : {}".format(self.bond_probability))
                    print("Bond Probability : {}".format(bond_probability))
                    print("Iterations       : {}".format(i))
                    print("Energy           : {}".format(self.total_energy))
                    print("Magnetization    : {}".format(self.magnetization))
                    print("x/y    : {}/{}".format(self.rij, self.kolom))

            if self.handler is not None and self.h5path is not None:
                self.handler.append(i, self.h5path+'iterations', dtype='int64')
                self.handler.append(self.total_energy, self.h5path+'energy')
                self.handler.append(self.magnetization, self.h5path+'magnetization')

            i += 1

    def printlattice(self):
        """ Prints the lattice. """
        g = self.grid
        str_ising = np.empty(g.size, dtype='int8').reshape(g.shape)
        str_ising[ np.where(g == 1) ] = 1 
        str_ising[ np.where(g == -1) ] = 8 
        print(str_ising)

def get_arguments():
    """
    To add arguments, call: parser.add_argument
    
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--iterations', default=100000, type=int,
                        help="Number of iterations, default: 100000") 
    parser.add_argument('-T', '--temperature', default=0.001, type=float,
                        help="The Temperature") 
    parser.add_argument('-b', '--bfield', default=0.00, type=float,
                        help="Uniform external magnetic field, default: 0") 
    parser.add_argument('-y', default=40,type=int, help="number of columns") 
    parser.add_argument('-x', default=40,type=int, help="number of rows") 
    parser.add_argument('-p', '--printit', default=None, type=int,
                        help="Print lattice every p flips") 
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--mode', default='metropolis', choices=['metropolis','wolff'],
                        help="Evolve with this algorithm.") 
   
    parser.add_argument('--full', action='store_true', help="Set lattice size\
                        such that it fills the terminal screen")
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    screensize = shutil.get_terminal_size(fallback=(80, 80))
    np.set_printoptions(threshold=np.nan, linewidth= 340)
    print(args)
    main()

