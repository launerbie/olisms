#!/usr/bin/env python

import sys
import argparse
import multiprocessing as mp
import time
import os
import logging
import hashlib
import numpy
from ising import Ising
from blessings import Terminal
from ext.progressbar import ProgressBar
from ext.hdf5handler import HDF5Handler
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

    tasks = []

    for job in jobs:
        for index, T in enumerate(numpy.linspace(int(cfg[job]['mintemp']),
                                                 int(cfg[job]['maxtemp']),
                                                 int(cfg[job]['steps']))):
            task = {
                    "algorithm":   str(cfg[job]['algorithm']),
                    "shape":       tuple(int(i) for i in cfg[job]['shape'].split('x')),
                    "aligned":     bool(cfg[job]['aligned']),
                    "mcs":         int(cfg[job]['mcs']),
                    "skip_n_steps":int(cfg[job]['skip_n_steps']),
                    "saveinterval":int(cfg[job]['saveinterval']),
                    "temperature": T,
                    "filename":    os.path.normpath(cfg[job]['filename']),
                    "h5path":      "/"+"sim_"+str(index).zfill(4)+"/",
                    "steps":   int(cfg[job]['steps']),
                   }

            tasks.append(task)

    logging.debug(tasks)
    return tasks

def hash_it(a):
    h = hashlib.sha256()
    h.update(str(a).encode('UTF-8'))
    hash_ = h.hexdigest()
    return hash_
    
def worker(tasks_queue, done_queue):

    for task in iter(tasks_queue.get, 'STOP'):
        process_id = int((mp.current_process().name)[-1]) #find nicer way
        writer = Writer((0, process_id), TERM)

        
        with Pbar(task, writer) as bar:
            with HDF5Handler(filename='/tmp/ising_data/'+hash_it(task)+'.hdf5') as h:
                isingsim = Ising(shape=task['shape'], sweeps=task['mcs'], 
                                 temperature=task['temperature'], 
                                 aligned=task['aligned'],
                                 mode=task['algorithm'], handler=h,
                                 saveinterval=task['saveinterval'],
                                 skip_n_steps=task['skip_n_steps'])
                isingsim.evolve(pbar=bar)

        timestamp = time.strftime("%c")
        job_report = "T={} time:{} (process {})".format(task["temperature"],
                                                        timestamp,
                                                        process_id)
        logging.info(job_report)
        done_queue.put(job_report)

def main():
    """
    Missing docstring
    """
    tasks_queue = mp.Queue()
    done_queue = mp.Queue()

    processpool = []
    for i in range(ARGS.nr_workers):
        p = mp.Process(target=worker, args=(tasks_queue, done_queue)).start()
        processpool.append(p)

    tasks = parsecfg(ARGS.config)
   
    with open('task_hashes.txt', 'w') as f:
        for t in tasks:
            f.write(str(t)+" : "+hash_it(t)+'\n')

    for t in tasks:
        tasks_queue.put(t)

    jobswriter = CompletedJobsWriter(TERM, (5,7)) #TODO: unhardcode
    for i in range(len(tasks)):
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
    def __init__(self, task, writer):
        self.description = "T:"+str(task["temperature"])+" "
        self.pbar = ProgressBar(widgets=drawwidget(self.description),
                                maxval=task["mcs"], fd=writer)

    def __enter__(self):
        self.pbar.start()
        return self.pbar

    def __exit__(self, exc_type, exc_val, traceback):
        self.pbar.finish()
        return False


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


