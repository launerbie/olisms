#!/usr/bin/env python

import sys
import argparse
import multiprocessing as mp
import time
import h5py
import os
import random
import logging
import hashlib
import numpy
from blessings import Terminal
from ext.progressbar import ProgressBar
from misc import drawwidget
"""
In mp_runsim.py the simulations are set up by reading a configfile.
The simulations are then processed by an x number of workers.
"""

def parsecfg(configfile):
    """
    Parse your configfile here.

    configfile: str
        Filepath to config file.

    returns: a list of Simulation instances.
    """

    interpolation = configparser.ExtendedInterpolation()
    cfg = configparser.ConfigParser(interpolation=interpolation)
    cfg.read(configfile)

    jobs = [s for s in cfg.sections() if s.startswith('job')]

    logging.debug(cfg.sections())
    logging.debug(jobs)

    sims = []

    for job in jobs:
        job_args = dict(algorithm = cfg[job]['algorithm'],
                        shape = tuple(cfg[job]['shape'].split('x')),
                        mcs = int(cfg[job]['mcs']),
                        saveinterval = cfg[job]['saveinterval'],
                        skip_n_steps = cfg[job]['skip_n_steps'],
                        minT = int(cfg[job]['mintemp']),
                        maxT = int(cfg[job]['maxtemp']),
                        steps = int(cfg[job]['steps']),
                        filename = os.path.normpath(cfg[job]['filename']),
                       )

        #logging.debug(job_args)
        sims.extend(split_job_into_sims(job_args))

    logging.debug(sims)
    return sims

def split_job_into_sims(job):
    simulations = []
    for index, T in enumerate(numpy.linspace(job['minT'],
                                             job['maxT'],
                                             job['steps'])):
        h5path = "/"+"sim_"+str(index).zfill(4)+"/"
        sim_kwargs = dict(shape=job['shape'],
                          mcs=job['mcs'],
                          temperature=T,
                          saveinterval=job['saveinterval'],
                          skip_n_steps=job['skip_n_steps'],
                          filename=os.path.normpath(job['filename']),
                          algorithm=job['algorithm'],
                          h5path=h5path,
                          handler="somehandler" #TODO;fix
                         )
        sim = Simulation(**sim_kwargs)
        simulations.append(sim)
    return simulations

class Simulation(object):
    """ Container for holding simulation parameters. """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        aligned : bool
            Set an initial grid with all spins aligned.

        temperature: float

        mcs : int
            Number of Monte-Carlo steps.
            For Metropolis, 1 MCS = 1 sweep.
            For Wolff, 1 MCS = 1 cluster flip (even though this is not
            actually correct)

        shape: tuple
            Shape of the lattice.

        algorithm: str
            Either 'metropolis' or 'wolff'
        """

        self.label = kwargs.get('label', 'unknown')
        self.aligned = kwargs.get('aligned', False)

        try:
            self.algorithm = kwargs['algorithm']
            self.temperature = kwargs['temperature']
            self.shape = kwargs['shape']
            self.mcs = kwargs['mcs']

#            self.handler =  kwargs['handler'] #what to do here?
            self.filename =  kwargs['filename']
            self.h5path = kwargs['h5path']
        except KeyError as e:
            print(e)
            raise Exception

    def __str__(self):
        return str(self.temperature)

    def __repr__(self):
        subs = (self.temperature, self.shape, self.h5path)
        return "T:{} shape:{} h5path:{}".format(*subs)

class FakeIsing(object):
    """ To be replaced by olisms.ising.Ising """

    def __init__(self, simulation, handler):
        assert isinstance(simulation, Simulation)
        self.params = simulation
        self.handler = handler

    def start(self, pbar):
        """ Start simulation, but for now let's just calculate some
        hashes."""
        string = str(self.params.label).encode('UTF-8')
        h = hashlib.sha256()
        h.update(string)

        for i in range(self.params.mcs):
            pbar.update(i)
            for j in range(200):
                hash_ = h.hexdigest()
                self.handler.put(hash_, "hash")
        return hash_

class Writer(object):
    """ Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """
    def __init__(self, location, term):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.term = term
        self.location = location

    def write(self, string):
        with self.term.location(*self.location):
            print(string)

class CompletedJobsWriter(object):
    """
    Merely a container for holding:
        1) the location in the terminal at which to print a list of
           completed jobs.
        2) a list of previously completed jobs, that will be no longer than
           the 'available' terminal space.
    """
    def __init__(self, terminal, location):
        """
        terminal: blessings.Terminal() instance
        location: tuple containing (x,y) terminal coordinates
        """
        self.width = terminal.width
        self.height = terminal.height
        self.maxheight = self.height - location[1] - 3
        self.location = location
        self.terminal = terminal
        self.lines = list()

    def print_line(self, line):
        """
        Prints a list of completed jobs (self.lines) to the terminal with the
        most recent completed job at the top.

        line : str
            Some job completion message.

        """
        if len(self.lines) == self.maxheight:
            self.lines.pop()
            self.lines = [line] + self.lines
        else:
            self.lines = [line] + self.lines
            assert len(self.lines) <= self.maxheight

        for i, line in enumerate(self.lines):
            with self.terminal.location(self.location[0], self.location[1]+i):
                print(line)

class Pbar(object):
    def __init__(self, sim, writer):
        self.description = str(sim.temperature) + " "
        self.pbar = ProgressBar(widgets=drawwidget(self.description),
                                maxval=sim.mcs, fd=writer)

    def __enter__(self):
        self.pbar.start()
        return self.pbar

    def __exit__(self, exc_type, exc_val, traceback):
        self.pbar.finish()
        return False

class FileStates(object):
    """ Keeps track of hdf5 files that are open."""

    def __init__(self, task):
        self.openfiles = {}

    def register_task(self, task):

        #filename = task['hdf5filename']
        filename = task.filename

        if filename in self.openfiles:
            hdf5handler = self.openfiles[filename]['handler']

        else:
            hdf5handler = HDF5Handler.open(filename)
            total = task['total_tasks']
            self.openfiles.update({filename:{'handler':hdf5handler,\
                                             'countdown':total}})

        return hdf5handler

    def completed_job(self, task):
        #Bepaal of de voltooide taak de 'hekkensluiter' is.
        #Zo ja, sluit bestand. Zo nee, doe niks.

        #filename = task['hdf5filename']
        filename = task.filename

        self.openfiles[filename]['countdown'] -= 1

        if self.openfiles[filename]['countdown'] == 0:
            handler = self.openfiles[filename]['handler']
            handler.close()
    
            self.openfiles.pop(filename)

def worker(tasks_queue, done_queue, filestates):
    """ 
    - pull task from tasks_queue
    - register task at FileStates, which will give you a handler.
    - pass handler to FakeIsing and call .start()
    - register task completion at FileStates
    """

    for sim in iter(tasks_queue.get, 'STOP'):
        process_id = int((mp.current_process().name)[-1]) #find nicer way
        writer = Writer((0, process_id), TERM)

        handler = filestates.register_task() #TODO
        

        with Pbar(sim, writer) as bar:
            isingsim = FakeIsing(sim, handler)
            isingsim.start(pbar=bar)

        timestamp = time.strftime("%c")

        job_report = "T={} time:{}".format(sim.temperature, timestamp)
        logging.info(job_report)
        done_queue.put(job_report)

def main():
    """
    Missing docstring
    """
    tasks_queue = mp.Queue()
    done_queue = mp.Queue()
    filestates = FileStates()

    processpool = []
    for i in range(ARGS.nr_workers):
        p = mp.Process(target=worker, args=(tasks_queue, done_queue, filestates)).start()
        processpool.append(p)

    sims = parsecfg(ARGS.config)
    for sim in sims:
        tasks_queue.put(sim)

    jobswriter = CompletedJobsWriter(TERM, (5,7)) #TODO: unhardcode
    for i in range(len(sims)):
        jobswriter.print_line(done_queue.get())

    for i in range(ARGS.nr_workers):
        tasks_queue.put('STOP')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help="Config file")
    parser.add_argument('--logfile', default='log_foo', help="logfile")
    parser.add_argument("--workers", dest='nr_workers', default=4,
                        type=int, help="Number of workers")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    if sys.version_info < (3, 3):
        s = "Running with Python {}, but Python 3.3 or greater is required."
        print(s.format(sys.version_info[:2]))
        exit()
    else:
        import configparser

    ARGS = get_arguments()

    logging.basicConfig(filename=ARGS.logfile, level=logging.DEBUG,)
    logging.info('START LOG FILE : ' + time.strftime("%c"))
    TERM = Terminal()
    print(TERM.clear())
    main()
    logging.info('END LOG FILE : ' + time.strftime("%c"))


