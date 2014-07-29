#!/usr/bin/env python
import time
import numpy as np
from itertools import permutations
from pprint import pprint
import subprocess
np.set_printoptions(threshold=np.nan, linewidth= 300)

def prod( iterable ):
    p = 1
    for n in iterable:
        p *= n
    return p

class Ising(object):
    def __init__(self, shape, sweeps, temperature=10, aligned=False, 
                 mode='metropolis', handler=None, h5path=None, 
                 saveinterval=1, skip_n_steps=0):
        """ 
        Parameters
        ----------
        shape : lattice shape 
        sweeps : total number of sweeps to perform
        temperature: temperature 
        aligned: create grid with all spins in the same direction
        mode : algorithm to use. Choices are: ['metropolis', 'wolff']
        handler: HDF5Handler instance
        h5path: unix-style path, used as address in the hdf5 file
        saveinterval: interval (in sweeps) at which data is saved to hdf5
        
        TODO:
        skip_n_steps: skip n steps before saving data 
    
        """
        self.mode = mode
        self.shape = tuple(shape)
        self.dimension = len(shape)
        self.temperature = temperature
        self.handler = handler 

        self.h5path = h5path 
        self.sweeps = sweeps
        self.saveinterval = saveinterval
        self.skip_n_steps = skip_n_steps

        self.ptable = self.make_probability_table()

        self.lattice_size = prod(self.shape)

        if mode == 'metropolis':
            self.evolve = self.evolve_metropolis
        elif mode == 'wolff':
            self.evolve = self.evolve_wolff
        else:
            raise ValueError("Unknown mode")

        if aligned:
            self.grid = np.ones(self.lattice_size, dtype=bool)
        else:
            grid = np.random.randint(2, size=self.lattice_size)
            self.grid = np.array(grid, dtype=bool)

        self.neighbor_table = self.make_neighbor_table()
        self.dE_map = self.make_dE_map()
        self.energy_map = self.make_energy_map()
        
        # save simulation parameters here 
        if (self.handler and self.h5path):
            self.writehdf5 = True
            self.handler.append(np.array(self.shape), self.h5path+'shape')
            self.handler.append(self.temperature, self.h5path+'temperature')
            self.handler.append(self.lattice_size, self.h5path+'lattice_size')
            self.handler.append(self.saveinterval, self.h5path+'saveinterval')
            self.handler.append(np.array(self.grid, dtype='int8'), 
                                self.h5path+'initgrid', dtype='int8')

            try:
                commit = subprocess.check_output(["git", "rev-parse","HEAD"])
            except:
                commit = "Unknown"

            #consider numpy.string_(commit)
            self.handler.file.attrs['commit'] = commit
            #consider numpy.string_(mode)
            self.handler.file.attrs['mode'] = mode

        else:
            self.writehdf5 = False

    #boolean array version
    @property
    def magnetization(self):
        M = 2*self.grid.sum() - len(self.grid)
        return M

    def make_probability_table(self):
        if self.dimension == 2:
            delta_energies = [-8, -4, 0, 4, 8] 
        elif self.dimension == 3:
            delta_energies = [-12, -8, -4, 0, 4, 8, 12]  
        else:
            s = "No probability table for lattice dimension "
            raise ValueError(s+"{}".format(self.dimension))

        ptable = dict() 
        for dE in delta_energies:
            ptable.update({dE:np.exp(-dE/self.temperature)}) 
        return ptable
   
    def make_neighbor_table(self):
        """
        Returns a dictionary where the keys are the sites and the values
        are the neighbors. So for a 4x4 lattice we have:

            {0: (1, 15, 4, 12),
             1: (2, 0, 5, 13),
             2: (3, 1, 6, 14),
             3: (4, 2, 7, 15),
             4: (5, 3, 8, 0),
             5: (6, 4, 9, 1),
             6: (7, 5, 10, 2),
             7: (8, 6, 11, 3),
             8: (9, 7, 12, 4),
             9: (10, 8, 13, 5),
             10: (11, 9, 14, 6),
             11: (12, 10, 15, 7),
             12: (13, 11, 0, 8),
             13: (14, 12, 1, 9),
             14: (15, 13, 2, 10),
             15: (0, 14, 3, 11)}

        """
        nbr_table_helical = dict()                              
        for site in range(self.lattice_size):
            nbr_table_helical.update({site:self.nn_helical_bc_2D(site)})
        return nbr_table_helical

    def make_energy_map(self):
        '''
        for site in range(self.lattice_size):
            below, above, right, left = self.neighbor_table[site]
            key = (bool(g[site]), (bool(g[right]), bool(g[below])) ) 
            #dE = self.energy_map[key] 
            #energy = energy + g[site]*( g[right] + g[below] )
            energy = energy + self.energy_map[key]


        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.
        '''
        energy_map = dict()

        #possible below and right neighbors
        config1 = (True, True)   
        config2 = (True, False)  
        config3 = (False, False) 

        for perm in set(permutations(config1)):
            energy_map.update({(True,perm):2})
            energy_map.update({(False,perm):-2})

        for perm in set(permutations(config2)):
            energy_map.update({(True,perm):0})
            energy_map.update({(False,perm):0})

        for perm in set(permutations(config3)):
            energy_map.update({(True,perm):-2})
            energy_map.update({(False,perm):2})
   
        return energy_map

    def make_dE_map(self):
        
        dE_map = dict()

        #possible neighbors
        config1 = (True,True,True,True)     
        config2 = (True,True,True,False)    
        config3 = (True,True,False,False)   
        config4 = (True,False,False,False) 
        config5 = (False,False,False,False)

        #CHECK THESE!
        for perm in set(permutations(config1)):
            dE_map.update({(True,perm):8})
            dE_map.update({(False,perm):-8})

        for perm in set(permutations(config2)):
            dE_map.update({(True,perm):4})
            dE_map.update({(False,perm):-4})

        for perm in set(permutations(config3)):
            dE_map.update({(True,perm) :0})
            dE_map.update({(False,perm) :0})

        for perm in set(permutations(config4)):
            dE_map.update({(True,perm):-4})
            dE_map.update({(False,perm):4})

        for perm in set(permutations(config5)):
            dE_map.update({(True,perm):-8})
            dE_map.update({(False,perm):8})
   
        return dE_map

    def nn_helical_bc_2D(self, site):                                                               
        """                                                                                   
        site: int                                                                             
        L: int                                                                                
                                                                                              
        Here i,j,k,l are the nearest neighbor indices for an ndarray of                       
        shape (1,) with helical boundary conditions.                                          
                                                                                              
                                                                                              
        So for example, here the neighbors of site=9 are shown:                               
                                                                                              
                                                                                              
                                  above                                                       
                              ------------|                                                   
                             \|/          |                                                   
        ----------------------------- L ----- R -----------------------------                 
        | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |                 
        ---------------------------------------------------------------------                 
                                          |          /|\                                      
                                          |------------                                       
                                             below                                            
                                                                                              
                                                                                              
         Which you can think of as a 2D lattice that looks like this:                         
                                                                                              
                                                                                              
                                            -------------                                     
          1   2   3   4   5   6   7   8   9 | 1 | 2 | 3 |                                     
                                 ------------------------                                     
          4   5   6   7   8   9 | 1 | 2 | 3 | 4 | 5 | 6 |                                     
                     -------------------------------------                                    
          7   8   9 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |                                     
        ------------|------------------------------------                                     
        | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 1 | 2 | 3 |                                     
        ------------|------------------------------------                                     
        | 4 | 5 | 6 | 7 | 8 | 9 | 1 | 2 | 3 | 4 | 5 | 6 |                                     
        ------------|------------------------------------                                     
        | 7 | 8 | 9 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |                                     
        ------------|------------------------------------                                     
        | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 1   2   3                                       
        ------------|------------------------                                                 
        | 4 | 5 | 6 | 7 | 8 | 9 | 1   2   3   4   5   6                                       
        ------------|------------                                                             
        | 7 | 8 | 9 | 1   2   3   4   5   6   7   8   9                                       
        -------------                                                                         
                                                                                              
                                                                                              
        """ 
        L = self.shape[0] 
        
        i = (site+1) % L**2                                                                      
        j = (site-1) % L**2                                                                      
        k = (site+L) % L**2                                                                      
        l = (site-L) % L**2                                                                      
        return i,j,k,l    

    def calc_energy(self): 
        """
        Function that iterates through the ising array and calculates product 
        of its value with right and lower neighbor. 
        """
        g = self.grid
        energy = 0

        for site in range(self.lattice_size):
            below, above, right, left = self.neighbor_table[site]
            key = (bool(g[site]), (bool(g[right]), bool(g[below])) ) 
            energy = energy + self.energy_map[key]
        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.

    def delta_energy(self, site):
        """  Returns dE = E2 - E1  (right??)     """
        g = self.grid
        below, above, right, left = self.neighbor_table[site]
        key = (bool(g[site]), (bool(g[below]), bool(g[above]), bool(g[right]), bool(g[left])) ) 
        dE = self.dE_map[key] 
        return dE 

        
    def evolve_metropolis(self, pbar, sleep=0):
        """
        Evolve it using Metropolis.
        """
        def sweep():
            for i in range(self.lattice_size):
                #if i % 1000 ==0:
                #    tempgrid = np.array(self.grid.reshape(self.shape[0],
                #                        self.shape[1]),dtype='int8')
                #    print(tempgrid)
                site = np.random.randint(0, self.lattice_size) #get this in chunks
                dE = self.delta_energy(site) 
                if dE <= 0: 
                    self.grid[site] = -self.grid[site]
                elif np.random.ranf() <= self.ptable[dE]:
                    self.grid[site] = -self.grid[site]
         

        for s in range(self.sweeps):
            pbar.update(s)
            sweep()
            
            if s % self.saveinterval == 0 and s >= self.skip_n_steps:
                if self.writehdf5:
                    self.handler.append(s, self.h5path+'sweep', dtype='int16')
                    #recalculating total energy at intervals likely faster 
                    #than continously updating total energy?
                    self.handler.append(self.calc_energy(), self.h5path+'energy')
                    self.handler.append(self.magnetization, self.h5path+'magnetization')

        self.handler.append(self.grid, self.h5path+'finalstate')


    def evolve_wolff(self, pbar, sleep=0):
        """
        Ewolve it using Wolff's algorithm.
        """
        g = self.grid

        J = 1
        kB = 1
        bond_probability = 1 - np.exp(-2*J/(kB*self.temperature))

        self.i=0

        while self.i < self.sweeps:
            pbar.update(self.i)

            cluster = list() 
            perimeter = list()

            seed = np.random.randint(0, self.lattice_size)
            seed_spin = g[seed]
            cluster.append(seed)

            nbrs = self.neighbor_table[seed]

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

                        nbrs = self.neighbor_table[site_to_test]
                        for nbr in nbrs:
                            if nbr not in cluster and nbr not in perimeter:
                                perimeter.append(nbr)

            #flip cluster
            g[np.array(cluster)] = -seed_spin

            if self.i % self.saveinterval == 0 and self.i >= self.skip_n_steps:
                if self.writehdf5:
                    self.handler.append(self.i, self.h5path+'clusterflip', dtype='int16')
                    self.handler.append(self.calc_energy(), self.h5path+'energy')
                    self.handler.append(self.magnetization, self.h5path+'magnetization')

            self.i += 1

        self.handler.append(self.grid, self.h5path+'finalstate')

    def print_sim_parameters(self):

        sweeps = self.sweeps
        Lx = self.shape[0] 
        Ly = self.shape[1]
        N = Lx*Ly
        saveinterval_in_iterations = N*self.saveinterval

        total_iters = sweeps * Lx * Ly 

        try:
            Lz = self.shape[2]
            N = Lx*Ly*Lz
            total_iters = sweeps * Lx * Ly * Lz
        except (IndexError, NameError):
            pass

        if self.mode == 'metropolis':
            s = """
            h5path             : {}   
            Algorithm          : {}  
            Lattice Shape      : {}  
            Lattice Size       : {}
            Temperature        : {}   
            Sweeps to perform  : {} (1 sweep = {} iterations)
            Total Iterations   : {} ({} * {} * {}) 
            Saving state every : {} sweeps (every {} iterations)
            """.format(self.h5path, self.mode, self.shape, N, self.temperature,
                       sweeps, N ,total_iters, sweeps, Lx, Ly, 
                       self.saveinterval, saveinterval_in_iterations)

        elif self.mode == 'wolff':
            s = """
            h5path             : {}   
            Algorithm          : {}  
            Lattice Shape      : {}  
            Lattice Size       : {}
            Temperature        : {}   
            Cluster flips      : {} 
            Saving state every : {} cluster flips
            """.format(self.h5path, self.mode, self.shape, N, self.temperature,
                       sweeps, self.saveinterval)
        print(s)
        #TODO
        #3D version

