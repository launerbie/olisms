#!/usr/bin/env python

import numpy as np
import os
import ext.progressbar as progressbar
from itertools import permutations

""" misc.py - Place some often reused functions here """

def drawwidget(discription, ljust=20):
    """ Formats the progressbar. """
    widget = [discription.ljust(ljust), progressbar.Percentage(), ' ',
              progressbar.Bar(marker='#', left='[', right=']'), ' ',
              progressbar.ETA()]
    return widget

def product(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p

def get_basename(filepath):
    """ If filepath = 'some_path/myfile.hdf5', then this returns 'myfile'. """
    basename = os.path.basename(filepath)
    name = os.path.splitext(basename)[0]
    return name

def acf(ndarr, length=None):
    """
    Return an ndarray with normalized correlation coefficients corresponding
    with lag elements given by: range(1, length).
    The range starts with 1 because the correlation for lag=0 is infinity.

    Parameters
    ----------
    ndarr : ndarray of shape (1,)
        The signal (typically time-series data) for which the autocorrelation
        function C(lag) needs to calculated.

    length : int
        End of the interval range(1, length). The normalized correlation
        coefficients are calculated for this range and returned as an ndarray.
        #TODO: or optional array of lag elements [dt1, dt2, dt3,...]

    See:

    https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

    for the definition of the correlation matrix.

    The correlation coefficients are returned by np.corrcoef. See for details:
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html

    """
    if not length:
        length = len(ndarr) - 1

    coeffs = [np.corrcoef(ndarr[:-dt], ndarr[dt:])[0, 1] \
              for dt in range(1, length)]
    result = np.array([1]+ coeffs)
    return result


def make_energy_map_2D():
    """
    This functions returns this dictionary:
    {(False, (False, False)): 2,
     (False, (False, True)): 0,
     (False, (True, False)): 0,
     (False, (True, True)): -2,
     (True, (False, False)): -2,
     (True, (False, True)): 0,
     (True, (True, False)): 0,
     (True, (True, True)): 2}
    """
    energy_map = dict()

    #possible below and right neighbors
    config1 = (True, True)
    config2 = (True, False)
    config3 = (False, False)

    for perm in set(permutations(config1)):
        energy_map.update({(True, perm):2})
        energy_map.update({(False, perm):-2})

    for perm in set(permutations(config2)):
        energy_map.update({(True, perm):0})
        energy_map.update({(False, perm):0})

    for perm in set(permutations(config3)):
        energy_map.update({(True, perm):-2})
        energy_map.update({(False, perm):2})

    return energy_map

def make_energy_map_3D():
    """
    This functions returns this dictionary:
    {(False, (False, False, False)): 3,
     (False, (False, False, True)): 1,
     (False, (False, True, False)): 1,
     (False, (True, False, False)): 1,
     (False, (True, True, Flase)): -1,
     (False, (True, False, True)): -1,
     (False, (False, True, True)): -1,
     (False, (True, True, True)): -3,
     (True, (False, False, False)): -3,
     (True, (False, False, True)): -1,
     (True, (False, True, False)): -1,
     (True, (True, False, False)): -1,
     (True, (True, True, Flase)): 1,
     (True, (True, False, True)): 1,
     (True, (False, True, True)): 1,
     (True, (True, True, True)): 3,
    """
    energy_map = dict()

    #possible below, right and front neighbors
    config1 = (True, True, True)
    config2 = (True, True, False)
    config3 = (True, False, False)
    config4 = (False, False, False)

    for perm in set(permutations(config1)):
        energy_map.update({(True,perm):3})
        energy_map.update({(False,perm):-3})

    for perm in set(permutations(config2)):
        energy_map.update({(True,perm):1})
        energy_map.update({(False,perm):-1})

    for perm in set(permutations(config3)):
        energy_map.update({(True,perm):-1})
        energy_map.update({(False,perm):1})

    for perm in set(permutations(config4)):
        energy_map.update({(True,perm):-3})
        energy_map.update({(False,perm):3})

    return energy_map

def make_delta_energy_map_2D():
    """
    {(False, (False, False, False, False)): 8,
     (False, (False, False, False, True)): 4,
     (False, (False, False, True, False)): 4,
     (False, (False, False, True, True)): 0,
     (False, (False, True, False, False)): 4,
     (False, (False, True, False, True)): 0,
     (False, (False, True, True, False)): 0,
     (False, (False, True, True, True)): -4,
     (False, (True, False, False, False)): 4,
     (False, (True, False, False, True)): 0,
     (False, (True, False, True, False)): 0,
     (False, (True, False, True, True)): -4,
     (False, (True, True, False, False)): 0,
     (False, (True, True, False, True)): -4,
     (False, (True, True, True, False)): -4,
     (False, (True, True, True, True)): -8,
     (True, (False, False, False, False)): -8,
     (True, (False, False, False, True)): -4,
     (True, (False, False, True, False)): -4,
     (True, (False, False, True, True)): 0,
     (True, (False, True, False, False)): -4,
     (True, (False, True, False, True)): 0,
     (True, (False, True, True, False)): 0,
     (True, (False, True, True, True)): 4,
     (True, (True, False, False, False)): -4,
     (True, (True, False, False, True)): 0,
     (True, (True, False, True, False)): 0,
     (True, (True, False, True, True)): 4,
     (True, (True, True, False, False)): 0,
     (True, (True, True, False, True)): 4,
     (True, (True, True, True, False)): 4,
     (True, (True, True, True, True)): 8}
    """

    delta_energy_map = dict()

    #possible neighbors
    config1 = (True, True, True, True)
    config2 = (True, True, True, False)
    config3 = (True, True, False, False)
    config4 = (True, False, False, False)
    config5 = (False, False, False, False)

    #CHECK THESE!
    for perm in set(permutations(config1)):
        delta_energy_map.update({(True, perm):8})
        delta_energy_map.update({(False, perm):-8})

    for perm in set(permutations(config2)):
        delta_energy_map.update({(True, perm):4})
        delta_energy_map.update({(False, perm):-4})

    for perm in set(permutations(config3)):
        delta_energy_map.update({(True, perm) :0})
        delta_energy_map.update({(False, perm) :0})

    for perm in set(permutations(config4)):
        delta_energy_map.update({(True, perm):-4})
        delta_energy_map.update({(False, perm):4})

    for perm in set(permutations(config5)):
        delta_energy_map.update({(True, perm):-8})
        delta_energy_map.update({(False, perm):8})

    return delta_energy_map

def make_delta_energy_map_3D():
    """ A straightforward addition to the 2D varaiant """
    dE_map = dict()

    #possible neighbors
    config1 = (True, True, True, True, True, True)
    config2 = (True, True, True, True, True, False)
    config3 = (True, True, True, True, False, False)
    config4 = (True, True, True, False, False, False)
    config5 = (True, True, False, False, False, False)
    config6 = (True, False, False, False, False, False)
    config7 = (False, False, False, False, False, False)

    #CHECK THESE!
    for perm in set(permutations(config1)):
        dE_map.update({(True,perm):12})
        dE_map.update({(False,perm):-12})

    for perm in set(permutations(config2)):
        dE_map.update({(True,perm):8})
        dE_map.update({(False,perm):-8})

    for perm in set(permutations(config3)):
        dE_map.update({(True,perm):4})
        dE_map.update({(False,perm):-4})

    for perm in set(permutations(config4)):
        dE_map.update({(True,perm) :0})
        dE_map.update({(False,perm) :0})

    for perm in set(permutations(config5)):
        dE_map.update({(True,perm):-4})
        dE_map.update({(False,perm):4})

    for perm in set(permutations(config6)):
        dE_map.update({(True,perm):-8})
        dE_map.update({(False,perm):8})

    for perm in set(permutations(config7)):
        dE_map.update({(True,perm):-12})
        dE_map.update({(False,perm):12})

    return dE_map

def probability_table(shape, temperature):
    """
    Returns a dictionary representing a probability table.
    shape : tuple
    temperature : float
    """
    dimension = len(shape)

    if dimension == 2:
        delta_energies = [-8, -4, 0, 4, 8]
    elif dimension == 3:
        delta_energies = [-12, -8, -4, 0, 4, 8, 12]
    else:
        message = "No probability table for lattice shape: "
        raise ValueError(message + "{}".format(shape))

    ptable = dict()
    for energy in delta_energies:
        ptable.update({energy:np.exp(-energy/temperature)})
    return ptable

def neighbor_table(shape):
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
    dimension = len(shape)
    size = product(shape)
    L = shape[0]

    nbr_table_helical = dict()

    if dimension == 2:
        nn_function = nn_helical_bc_2D
    elif dimension == 3:
        nn_function = nn_helical_bc_3D
    else:
        raise Exception("Unsupported dimension: {}".format(dimension))

    for site in range(size):
        nbr_table_helical.update({site:nn_function(site, L)})

    return nbr_table_helical


def nn_helical_bc_2D(site, width):
    """
    site: int
    width: int

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
    below = (site+1) % width**2
    above = (site-1) % width**2
    right = (site+width) % width**2
    left = (site-width) % width**2
    return below, above, right, left

def nn_helical_bc_3D(site, L):
    """
    Same idea as the 2D variant, just now it's some kind of hyperdougnut.
    """
    i = (site+1) % L**3
    j = (site-1) % L**3
    k = (site+L) % L**3
    l = (site-L) % L**3
    m = (site+L**2) % L**3
    n = (site-L**2) % L**3
    return i,j,k,l,m,n

def get_delta_energy_function(ising):
    dimension = len(ising.shape)
    g = ising.grid

    if dimension == 2:
        nbr_table = neighbor_table(ising.shape)
        delta_energy_map = make_delta_energy_map_2D()

        def delta_energy(site):
            below, above, right, left = nbr_table[site]
            key = (bool(g[site]), (bool(g[below]), bool(g[above]), bool(g[right]),
                                   bool(g[left])))
            delta_energy = delta_energy_map[key]
            return delta_energy

    elif dimension == 3:
        nbr_table = neighbor_table(ising.shape)
        delta_energy_map = make_delta_energy_map_3D()

        def delta_energy(site):
            below, above, right, left, front, back = nbr_table[site]
            key = (bool(g[site]), (bool(g[below]), bool(g[above]), bool(g[right]),
                   bool(g[left]), bool(g[front]), bool(g[back])) )
            delta_energy = delta_energy_map[key]
            return delta_energy

    else:
        raise Exception("Unsupported dimension: {}".format(dimension))

    return delta_energy


def get_calc_energy_function(ising):
    dimension = len(ising.shape)
    g = ising.grid
    size = product(ising.shape)

    if dimension == 2:
        nbr_table = neighbor_table(ising.shape)
        energy_map = make_energy_map_2D()

        def calc_energy():
            """ Returns the total energy """
            energy = 0
            for site in range(size):
                below, above, right, left = nbr_table[site]
                key = (bool(g[site]), (bool(g[right]), bool(g[below])))
                energy = energy + energy_map[key]
            return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.

    elif dimension == 3:
        nbr_table = neighbor_table(ising.shape)
        energy_map = make_energy_map_3D()

        def calc_energy():
            """ Returns the total energy """
            energy = 0
            for site in range(size):
                below, above, right, left, front, back = nbr_table[site]
                key = (bool(g[site]),
                       (bool(g[right]), bool(g[below]), bool(g[front])))
                energy = energy + energy_map[key]
            return -energy  # H = -J*SUM(nearest neighbors) Let op de -J.
    else:
        raise Exception("Unsupported dimension: {}".format(dimension))

    return calc_energy

def print_sim_parameters(ising):

    sweeps = ising.sweeps
    width = ising.shape[0]
    height = ising.shape[1]
    lattice_size = width * height
    saveinterval_in_iterations = lattice_size * ising.saveinterval

    total_iters = sweeps * lattice_size

    try:
        depth = ising.shape[2]
        lattice_size = width * height * depth
        total_iters = sweeps * lattice_size
    except (IndexError, NameError):
        pass

    if ising.mode == 'metropolis':
        simparams = """
        Algorithm          : {}
        Lattice Shape      : {}
        Lattice Size       : {}
        Temperature        : {}
        Sweeps to perform  : {} (1 sweep = {} iterations)
        Total Iterations   : {} ({} * {} * {})
        Saving state every : {} sweeps (every {} iterations)
        """.format(ising.mode, ising.shape, lattice_size,
                   ising.temperature,
                   sweeps, lattice_size, total_iters, sweeps, width, height,
                   ising.saveinterval, saveinterval_in_iterations)

    elif ising.mode == 'wolff':
        simparams = """
        Algorithm          : {}
        Lattice Shape      : {}
        Lattice Size       : {}
        Temperature        : {}
        Cluster flips      : {}
        Saving state every : {} cluster flips
        """.format(ising.mode, ising.shape, lattice_size,
                   ising.temperature, sweeps, ising.saveinterval)
    print(simparams)
    #TODO
    #3D version
