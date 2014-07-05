#!/usr/bin/env python

import shutil
import argparse
import numpy as np

def prod( iterable ):
    p = 1
    for n in iterable:
        p *= n
    return p

class Ising(object):
    def __init__(self, shape, temperature=10, aligned=False, mode='metropolis',
                 handler=None, h5path=None):
        """ 
        Parameters
        ----------
        shape : lattice shape 
        temperature: temperature 
        mode : algorithm to use. Choices are: ['metropolis', 'wolff']
        handler: HDF5Handler instance
        h5path: unix-style path, used as address in the hdf5 file
    
        """
        if mode == 'metropolis':
            self.evolve = self.evolve_metropolis
        elif mode == 'wolff':
            self.evolve = self.evolve_wolff
        else:
            raise ValueError("Unknown mode")


        self.shape = tuple(shape)
        self.dimension = len(shape)

        if self.dimension == 2:
            self.neighbors = self.neighbors_2D
            self.choose_site = self.choose_site_2D
            self.delta_energy = self.delta_energy_2D
            self.calc_energy = self.calc_energy_2D
        elif self.dimension == 3:
            self.neighbors = self.neighbors_3D
            self.choose_site = self.choose_site_3D
            self.delta_energy = self.delta_energy_3D
            self.calc_energy = self.calc_energy_3D
        else:
            raise ValueError("Unsupported dimension")


        self.grid = self.makegrid(aligned=aligned)
        self.temperature = temperature
        self.total_energy = self.calc_energy()
        self.handler = handler 
        self.h5path = h5path 
        self.ptable = self.make_probability_table()

        self.lattice_size = prod(self.shape)
        
        if (self.handler and self.h5path) is not None:
            self.handler.append(self.temperature, self.h5path+'temperature')
            self.handler.append(self.lattice_size, self.h5path+'lattice_size')
            self.handler.append(np.array(self.grid, dtype='int8'), 
                                self.h5path+'initgrid', dtype='int8')

    @property
    def magnetization(self):
        return self.grid.sum()

    def make_probability_table(self):
        if self.dimension == 2:
            delta_energies = [-8, -4, 0, 4, 8] 
        elif self.dimension == 3:
            delta_energies = [-12, -8, -4, 0, 4, 8, 12]  
        else:
            raise ValueError("No probability table for lattice dimension {}".format(self.dimension))

        ptable = dict() 
        for dE in delta_energies:
            ptable.update({dE:np.exp(-dE/self.temperature)}) 
        return ptable

    def choose_site_2D(self):
        site_x = np.random.randint(0, self.shape[0])
        site_y = np.random.randint(0, self.shape[1])
        return site_x, site_y

    def choose_site_3D(self):
        site_x = np.random.randint(0, self.shape[0])
        site_y = np.random.randint(0, self.shape[1])
        site_z = np.random.randint(0, self.shape[2])
        return site_x, site_y, site_z

    def makegrid(self, aligned=False):
        """ 
        If aligned = True, a grid of ones is returned.
        """
        if aligned:
            grid = np.ones(self.shape)
            return np.array(grid, dtype='int8')
        else:
            grid = np.random.choice([-1, 1], size=self.shape)
            return np.array(grid, dtype='int8')

    def neighbors_2D(self, site):
        i, j = site
        ROW = self.shape[0]
        COL = self.shape[1]

        left = (j + COL - 1) % COL
        right = (j + 1) % COL
        above = (i + ROW - 1) % ROW
        below = (i + 1) % ROW

        nbrs = (below, j), (above, j), (i, right), (i, left)
        return nbrs

    def neighbors_3D(self, site):
        i, j, k = site

        ROW = self.shape[0]
        COL = self.shape[1]
        DEP = self.shape[2]

        left = (j + COL - 1) % COL 
        right = (j + 1) % COL 
        above = (i + ROW - 1) % ROW
        below = (i + 1) % ROW

        front = (k + DEP - 1) % DEP
        back = (i + 1) % DEP

        nbrs = (below, j, k), (above, j, k), (i, right, k), (i, left, k), (i, j, back), (i, j, front)
        return nbrs

        
    #def neighbors(self, site):
    #    """
    #    Returns a list of sites which are the neighbors of 'site'.

    #    Return
    #    ------
    #    tuple: (below nbr), (above nbr), (right nbr), (left nbr)
    #    
    #    Here: "below", "above", "right" and "left" mean:

    #               j-1         j         j+1

    #            ---------------------------------
    #    i-1     |        |    above   |         |
    #            |        |            |         |
    #            ---------------------------------
    #     i      |  left  |    site    |  right  |
    #            |        |   (i, j)   |         |
    #            ---------------------------------
    #    i+1     |        |            |         |
    #            |        |    below   |         |
    #            ---------------------------------


    #    So if you consider the ndarray a=numpy.arange(9).reshape(3,3):

    #            >>> a
    #            array([[0, 1, 2],
    #                   [3, 4, 5],
    #                   [6, 7, 8]])
    #            >>> print(a[0,0], a[1,1], a[2,2])
    #            0 4 8

    #    Then the left neighbor of 4 is 3.

    #    """
    #    LR = self.rij - 1
    #    LK = self.kolom - 1
    #    i, j = site
    #    if not (i == 0 or i == LR or j == 0 or j == LK): # dan niet rand  
    #        nbrs = (i+1, j), (i-1, j), (i, j+1), (i, j-1)
    #    else: #dan rand
    #        if not ((i == 0 and (j == 0 or j == LK)) or\
    #                (i == LR and (j == 0 or j == LK))): #dan niet hoek
    #            if i == 0:
    #                nbrs = (i+1, j), (LR, j), (i, j+1), (i, j-1)
    #            elif i == LR:
    #                nbrs = (0, j), (i-1, j), (i, j+1), (i, j-1)
    #            elif j == 0:
    #                nbrs = (i+1, j), (i-1, j), (i, j+1), (i, LK)
    #            else:
    #                nbrs = (i+1, j), (i-1, j), (i, 0), (i, j-1)
    #        else: # dan hoek
    #            if (i == 0 and j == 0):
    #                nbrs = (i+1, j), (LR, j), (i, j+1), (i, LK)
    #            elif (i == 0 and j == LK):
    #                nbrs = (i+1, j), (LR, j), (i, 0), (i, j-1)
    #            elif (i == LR and j == 0):
    #                nbrs = (0, j), (i-1, j), (i, j+1), (i, LK)
    #            else:
    #                nbrs = (0, j), (i-1, j), (i, 0), (i, j-1)
    #    return nbrs


    def calc_energy_2D(self): 
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in column with 
        first in column (torus).

        """
        g = self.grid
        energy = 0

        for site, value in np.ndenumerate(g):
            below, above, right, left = self.neighbors(site)
            energy = energy + g[site]*( g[right] + g[below] )

        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.

    def calc_energy_3D(self):
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. Boundary conditions link 
        last entry in row with first in row and last entry in column with 
        first in column (torus).

        """
        g = self.grid
        energy = 0 
        for site, value in np.ndenumerate(g):
            below, above, right, left, back, front = self.neighbors(site)
            energy = energy + g[site]*( g[right] + g[below] + g[front])

        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.


    def delta_energy_2D(self, site):
        """
        Berekent verandering in energie als gevolg van het omdraaien van het 
        teken (spin flip) op de positie "site".

        """
        g = self.grid
        below, above, right, left = self.neighbors(site)
        d_energy = -2*(-g[site] * (g[below] + g[above] +g[right] +g[left]))
        return d_energy  


    def delta_energy_3D(self, site):
        """
        Berekent verandering in energie als gevolg van het omdraaien van het 
        teken (spin flip) op de positie "site".
        """
        g = self.grid
        below, above, right, left, back, front = self.neighbors(site)
        d_energy = -2*(-g[site] * (g[below] + g[above] + g[right] + g[left] + g[back] + g[front]))
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
            flipped = self.flip(probability, site) 
            
            if flipped and delta_e != 0:
                flipcount += 1 
                self.total_energy = self.total_energy + delta_e

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

            if self.handler is not None and self.h5path is not None:
                self.handler.append(i, self.h5path+'iterations', dtype='int64')
                self.handler.append(self.total_energy, self.h5path+'energy')
                self.handler.append(self.magnetization, self.h5path+'magnetization')

            i += 1


