#!/usr/bin/env python
import time
import numpy as np
import subprocess
from misc import product
from misc import make_energy_map
from misc import make_delta_energy_map
from misc import neighbor_table
from misc import probability_table

class Ising(object):
    """ Missing docstring """
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


        """
        self.mode = mode
        self.shape = tuple(shape)
        self.dimension = len(shape)

        if self.dimension == 2:
            self.width = shape[0]
            self.height = shape[1]
        elif self.dimension == 3:
            self.width = shape[0]
            self.height = shape[1]
            self.depth = shape[2]
        else:
            raise Exception("Unsupported dimension")

        self.temperature = temperature
        self.handler = handler

        self.h5path = h5path
        self.sweeps = sweeps
        self.saveinterval = saveinterval
        self.skip_n_steps = skip_n_steps

        self.ptable = probability_table(self.dimension, self.temperature)

        self.lattice_size = product(self.shape)

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

        self.nbr_table_br = neighbor_table(self.width, only_below_right=True)
        self.nbr_table = neighbor_table(self.width)
        self.delta_energy_map = make_delta_energy_map()
        self.energy_map = make_energy_map()

        # save simulation parameters here
        if self.handler and self.h5path:
            self.writehdf5 = True
            self.handler.append(np.array(self.shape), self.h5path+'shape')
            self.handler.append(self.temperature, self.h5path+'temperature')
            self.handler.append(self.lattice_size, self.h5path+'lattice_size')
            self.handler.append(self.saveinterval, self.h5path+'saveinterval')
            self.handler.append(np.array(self.grid, dtype='int8'),
                                self.h5path+'initgrid', dtype='int8')

            commit = subprocess.check_output(["git", "rev-parse", "HEAD"])

            #consider numpy.string_(commit)
            self.handler.file.attrs['commit'] = commit
            #consider numpy.string_(mode)
            self.handler.file.attrs['mode'] = mode

        else:
            self.writehdf5 = False

    @property
    def magnetization(self):
        """ Returns the total magnetization for a boolean array. """
        magnetization = 2*self.grid.sum() - len(self.grid)
        return magnetization

    def calc_energy(self):
        """
        Function that iterates through the ising array and calculates product
        of its value with right and lower neighbor.
        """
        g = self.grid
        energy = 0

        for site in range(self.lattice_size):
            below, right = self.nbr_table_br[site]
            key = (bool(g[site]), (bool(g[right]), bool(g[below])))
            energy = energy + self.energy_map[key]
        return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.

    def delta_energy(self, site):
        """  Returns delta_energy = E2 - E1  (right??)     """
        g = self.grid
        below, above, right, left = self.nbr_table[site]
        key = (bool(g[site]), (bool(g[below]), bool(g[above]), bool(g[right]),
                               bool(g[left])))
        delta_energy = self.delta_energy_map[key]
        return delta_energy


    def evolve_metropolis(self, pbar): #pbar shouldn't be mandatory argument
        """
        Evolve it using Metropolis.
        """
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
                    self.handler.append(sweep, self.h5path+'sweep',
                                        dtype='int16')
                    self.handler.append(self.calc_energy(),
                                        self.h5path+'energy')
                    self.handler.append(self.magnetization,
                                        self.h5path+'magnetization')

        self.handler.append(self.grid, self.h5path+'finalstate')


    def evolve_wolff(self, pbar):
        """
        Ewolve it using Wolff's algorithm.
        """
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
                    self.handler.append(flip, self.h5path+'clusterflip',
                                        dtype='int16')
                    self.handler.append(self.calc_energy(),
                                        self.h5path+'energy')
                    self.handler.append(self.magnetization,
                                        self.h5path+'magnetization')


        self.handler.append(self.grid, self.h5path+'finalstate')

    def print_sim_parameters(self):

        sweeps = self.sweeps
        width = self.shape[0]
        height = self.shape[1]
        lattice_size = width * height
        saveinterval_in_iterations = lattice_size * self.saveinterval

        total_iters = sweeps * lattice_size

        try:
            depth = self.shape[2]
            lattice_size = width * height * depth
            total_iters = sweeps * lattice_size
        except (IndexError, NameError):
            pass

        if self.mode == 'metropolis':
            simparams = """
            h5path             : {}
            Algorithm          : {}
            Lattice Shape      : {}
            Lattice Size       : {}
            Temperature        : {}
            Sweeps to perform  : {} (1 sweep = {} iterations)
            Total Iterations   : {} ({} * {} * {})
            Saving state every : {} sweeps (every {} iterations)
            """.format(self.h5path, self.mode, self.shape, lattice_size,
                       self.temperature,
                       sweeps, lattice_size, total_iters, sweeps, width, height,
                       self.saveinterval, saveinterval_in_iterations)

        elif self.mode == 'wolff':
            simparams = """
            h5path             : {}
            Algorithm          : {}
            Lattice Shape      : {}
            Lattice Size       : {}
            Temperature        : {}
            Cluster flips      : {}
            Saving state every : {} cluster flips
            """.format(self.h5path, self.mode, self.shape, lattice_size,
                       self.temperature, sweeps, self.saveinterval)
        print(simparams)
        #TODO
        #3D version

class IsingAnim(Ising):
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


