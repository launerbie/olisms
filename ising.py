#!/usr/bin/env python
import time
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
        aligned: create grid with all spins in the same direction
        mode : algorithm to use. Choices are: ['metropolis', 'wolff']
        handler: HDF5Handler instance
        h5path: unix-style path, used as address in the hdf5 file
    
        """
        self.shape = tuple(shape)
        self.dimension = len(shape)
        self.temperature = temperature
        self.handler = handler 
        self.h5path = h5path 
        self.ptable = self.make_probability_table()

        self.lattice_size = prod(self.shape)

        if mode == 'metropolis':
            self.evolve = self.evolve_metropolis
        elif mode == 'wolff':
            self.evolve = self.evolve_wolff
        else:
            raise ValueError("Unknown mode")


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


        if aligned:
            self.grid = np.ones(self.shape, dtype='int8')
        else:
            grid = np.random.choice([-1, 1], size=self.shape)
            self.grid = np.array(grid, dtype='int8')
        
        self.total_energy = self.calc_energy()
        
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
        """ 
        Flip 'site' with probability 'prob'.

        Parameters
        ----------
        prob: probability 
        site: tuple (e.g. (i,j) if 2D, (i,j,k) if 3D)
        """
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

        
    def evolve_metropolis(self, iterations, sleep=0):
        """
        Evolve it using Metropolis.
        """
        self.i = 0
        flipcount = 0
         
        while self.i < iterations:
            time.sleep(sleep)
            site = self.choose_site() 
            delta_e = self.delta_energy(site) 
            probability = self.ptable[delta_e]
            flipped = self.flip(probability, site) 
            
            if flipped and delta_e != 0:
                flipcount += 1 
                self.total_energy = self.total_energy + delta_e

            if self.handler is not None and self.h5path is not None:
                self.handler.append(np.array(site), self.h5path+'sites', dtype='int16')
                self.handler.append(self.i, self.h5path+'iterations', dtype='int64')
                self.handler.append(self.total_energy, self.h5path+'energy')
                self.handler.append(self.magnetization, self.h5path+'magnetization')

            self.i += 1


    def evolve_wolff(self, iterations, sleep=0):
        """
        Ewolve it using Wolff's algorithm.
        """
        g = self.grid

        J = 1
        kB = 1
        bond_probability = 1 - np.exp(-2*J/(kB*self.temperature))

        self.i=0
        while self.i < iterations:
            time.sleep(sleep)
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
                self.handler.append(self.i, self.h5path+'iterations', dtype='int64')
                self.handler.append(self.total_energy, self.h5path+'energy')
                self.handler.append(self.magnetization, self.h5path+'magnetization')

            self.i += 1


