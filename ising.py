#!/usr/bin/env python
import time
import numpy as np
import subprocess
from misc import product
from misc import neighbor_table
from misc import probability_table
from misc import get_delta_energy_function
from misc import get_calc_energy_function
from misc import print_sim_parameters as printer

class Ising(object):
    """ Missing docstring """
    def __init__(self, shape, sweeps, temperature=10, aligned=False,
                 algorithm='metropolis', handler=None,
                 saveinterval=1, skip_n_steps=0):
        """
        Parameters
        ----------
        shape : lattice shape
        sweeps : total number of sweeps to perform
        temperature: temperature
        aligned: create grid with all spins in the same direction
        algorithm : algorithm to use. Choices are: ['metropolis', 'wolff']
        handler: HDF5Handler instance
        saveinterval: interval (in sweeps) at which data is saved to hdf5
        """
        self.algorithm = algorithm
        self.shape = tuple(shape)
        self.lattice_size = product(self.shape)
        self.temperature = temperature

        self.sweeps = sweeps
        self.saveinterval = saveinterval
        self.skip_n_steps = skip_n_steps

        self.handler = handler

        if aligned:
            self.grid = np.ones(self.lattice_size, dtype=bool)
        else:
            grid = np.random.randint(2, size=self.lattice_size)
            self.grid = np.array(grid, dtype=bool)

        self.ptable = probability_table(self.shape, self.temperature)
        self.nbr_table = neighbor_table(self.shape)
        self.delta_energy = get_delta_energy_function(self)
        self.calc_energy = get_calc_energy_function(self)

        # save simulation parameters here
        if self.handler:
            self.writehdf5 = True
            self.handler.put(np.array(self.shape), 'shape')
            self.handler.put(self.temperature, 'temperature')
            self.handler.put(self.lattice_size, 'lattice_size')
            self.handler.put(self.saveinterval, 'saveinterval')
            self.handler.put(np.array(self.grid, dtype='int8'), 'initgrid',
                             dtype='int8')


            commit = subprocess.check_output(["git", "rev-parse", "HEAD"])

            #consider numpy.string_(commit)
            self.handler.file.attrs['commit'] = commit
            #consider numpy.string_(algorithm)
            self.handler.file.attrs['algorithm'] = algorithm

        else:
            self.writehdf5 = False

    @property
    def magnetization(self):
        """ Returns the total magnetization for a boolean array. """
        magnetization = 2*self.grid.sum() - len(self.grid)
        return magnetization

    def print_sim_parameters(self):
        printer(self)

    def set_grid(self, ndarr, spin_up=1, spin_down=-1):
        """ doc """
        g = ndarr.flatten()
        g[np.where(g == spin_up)] = True
        g[np.where(g == spin_down)] = False
        self.shape = ndarr.shape
        self.lattice_size = product(self.shape)
        self.grid = g
        self.delta_energy = get_delta_energy_function(self)
        self.calc_energy = get_calc_energy_function(self)

    def evolve(self, pbar):
        if self.algorithm == 'metropolis':
            self.evolve_metropolis(pbar)
        elif self.algorithm == 'wolff':
            self.evolve_wolff(pbar)
        else:
            raise ValueError("Unknown algorithm")

    def evolve_metropolis(self, pbar): #pbar shouldn't be mandatory argument
        """ Evolve it using Metropolis. """

        def do_sweep():
            for _ in range(self.lattice_size):
                site = np.random.randint(0, self.lattice_size)
                delta_energy = self.delta_energy(site)
                if delta_energy <= 0:
                    self.grid[site] = -self.grid[site]
                elif np.random.ranf() <= self.ptable[delta_energy]:
                    self.grid[site] = -self.grid[site]

        for sweep in range(self.sweeps):
            pbar.update(sweep)
            do_sweep()

            if sweep % self.saveinterval == 0 and sweep >= self.skip_n_steps:
                if self.writehdf5:
                    self.handler.put(sweep, 'sweep', dtype='int16')
                    self.handler.put(self.calc_energy(), 'energy')
                    self.handler.put(self.magnetization, 'magnetization')

        self.handler.put(self.grid, 'finalstate')

    def evolve_wolff(self, pbar):
        """ Ewolve it using Wolff's algorithm. """

        g = self.grid

        bond_probability = 1 - np.exp(-2/self.temperature)

        for flip in range(self.sweeps):
            pbar.update(flip)

            cluster = list()
            perimeter = list()

            seed = np.random.randint(0, self.lattice_size)
            seed_spin = g[seed]
            cluster.append(seed)

            nbrs = self.nbr_table[seed]

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

                        nbrs = self.nbr_table[site_to_test]
                        for nbr in nbrs:
                            if nbr not in cluster and nbr not in perimeter:
                                perimeter.append(nbr)

            #flip cluster
            g[np.array(cluster)] = -seed_spin

            if flip % self.saveinterval == 0 and flip >= self.skip_n_steps:
                if self.writehdf5:
                    self.handler.put(flip, 'clusterflip', dtype='int16')
                    self.handler.put(self.calc_energy(), 'energy')
                    self.handler.put(self.magnetization, 'magnetization')

        self.handler.put(self.grid, 'finalstate')


class IsingAnim(Ising):

    def evolve(self, *args, **kwds):
        if self.algorithm == 'metropolis':
            self.evolve_metropolis(*args, **kwds)
        elif self.algorithm == 'wolff':
            self.evolve_wolff(*args, **kwds)
        else:
            raise ValueError("Unknown algorithm")

    def evolve_metropolis(self, sleep=0):
        self.sweepcount = 0
        def sweep():
            for _ in range(self.lattice_size):
                site = np.random.randint(0, self.lattice_size)
                delta_energy = self.delta_energy(site)
                if delta_energy <= 0:
                    self.grid[site] = -self.grid[site]
                elif np.random.ranf() < self.ptable[delta_energy]:
                    self.grid[site] = -self.grid[site]


        for s in range(self.sweeps):
            time.sleep(sleep)
            sweep()
            self.sweepcount += 1


    def evolve_wolff(self, sleep=0):
        g = self.grid

        bond_probability = 1 - np.exp(-2/self.temperature)

        self.sweepcount = 0

        while self.sweepcount < self.sweeps:
            time.sleep(sleep)

            cluster = list()
            perimeter = list()

            seed = np.random.randint(0, self.lattice_size)
            seed_spin = g[seed]
            cluster.append(seed)

            nbrs = self.nbr_table[seed]

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

                        nbrs = self.nbr_table[site_to_test]
                        for nbr in nbrs:
                            if nbr not in cluster and nbr not in perimeter:
                                perimeter.append(nbr)

            #flip cluster
            g[np.array(cluster)] = -seed_spin

            self.sweepcount += 1


