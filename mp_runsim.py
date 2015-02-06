#!/usr/bin/env python

import h5py
import sys
import argparse
import multiprocessing as mp
import time
import os
import shutil
import logging
import hashlib
import numpy
import itertools
from ising import Ising
from blessings import Terminal
from ext.progressbar import ProgressBar
from ext.hdf5handler import HDF5Handler
from misc import drawwidget

"""
In mp_runsim.py the simulations are set up by reading a configfile.
The simulations are then processed by an x number of workers.
"""

def hash_it(a):
    h = hashlib.sha256()
    h.update(str(a).encode('UTF-8'))
    hash_ = h.hexdigest()
    return hash_


def worker(tasks_queue, done_queue):

    for task, hash_ in iter(tasks_queue.get, 'STOP'):
        process_id = int((mp.current_process().name)[-1]) #find nicer way
        writer = Writer((0, process_id), TERM)


        with Pbar(task, writer) as bar:
            with HDF5Handler(filename=TEMPDIR+hash_+'.hdf5') as h:
                isingsim = Ising(shape=task['shape'], sweeps=task['mcs'],
                                 temperature=task['temperature'],
                                 aligned=task['aligned'],
                                 algorithm=task['algorithm'], handler=h,
                                 saveinterval=task['saveinterval'],
                                 skip_n_steps=task['skip_n_steps'])
                isingsim.evolve(pbar=bar)

        timestamp = time.strftime("%d %b %Y %H:%M:%S")
        job_report = "T={0:.3f}  {1}".format(task["temperature"],timestamp)
        logging.info(job_report)
        done_queue.put(job_report)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help="Config file")
    parser.add_argument('-d', '--outputdir', required=True, help="Target\
                        directory for hdf5 files")
    parser.add_argument('-p', '--prefix', default="", help="adds [prefix] to filenames")
    parser.add_argument('-l', '--logfile', default='log_foo', help="logfile")
    parser.add_argument('-w', "--workers", dest='nr_workers', default=4,
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
    TEMPDIR = '/tmp/olisms/'

    if not os.path.exists(TEMPDIR):
        os.makedirs(TEMPDIR)
    else:
        shutil.rmtree(TEMPDIR)
        os.makedirs(TEMPDIR)

    logging.basicConfig(filename=ARGS.logfile, level=logging.DEBUG,)
    logging.info('START LOG FILE : ' + time.strftime("%c"))
    TERM = Terminal()
    print(TERM.clear())

    tasks_queue = mp.Queue()
    done_queue = mp.Queue()

    processpool = []
    for i in range(ARGS.nr_workers):
        p = mp.Process(target=worker, args=(tasks_queue, done_queue)).start()
        processpool.append(p)

    interpolation = configparser.ExtendedInterpolation()
    cfg = configparser.ConfigParser(interpolation=interpolation)
    cfg.read(ARGS.config)

    def job_to_tasks(job):
        tasks = []
        for index, T in enumerate(numpy.linspace(float(cfg[job]['mintemp']),
                                                 float(cfg[job]['maxtemp']),
                                                 int(cfg[job]['steps']))):
            task = {
                    "algorithm":   str(cfg[job]['algorithm']),
                    "shape":       tuple(int(i) for i in \
                                         cfg[job]['shape'].split('x')),
                    "aligned":     bool(cfg[job]['aligned']),
                    "mcs":         int(cfg[job]['mcs']),
                    "skip_n_steps":int(cfg[job]['skip_n_steps']),
                    "saveinterval":int(cfg[job]['saveinterval']),
                    "temperature": T,
                   }
            tasks.append(task)
        return tasks

    def job_to_hashes(job):
        tasks = job_to_tasks(job)
        hashes = [hash_it(task) for task in tasks]
        return hashes

    def unique_filename_from_job(job):
        identifier = "{algorithm}_{shape}_MCS{mcs}_si{saveinterval}_\
                      minT{mintemp}_maxT{maxtemp}_{steps}_\
                      {aligned}".format(**cfg[job])

        identifier_no_whitespace = identifier.replace(" ", "")
        return identifier_no_whitespace


    def filename_from_job(job):
        data_dir = ARGS.outputdir
        prefix = ARGS.prefix

        job_id = unique_filename_from_job(job)

        abs_path = "{data_dir}/{prefix}{job_id}.hdf5".format(**locals())
        return abs_path

    jobs = [job for job in cfg.sections() if job.startswith('job')]

    tasks_grouped_by_job = [job_to_tasks(job) for job in jobs]
    # tasks_grouped_by_job =  [ [t1, t2, t3], [t1, t2, t3], etc ]

    hashes_grouped_by_job = [job_to_hashes(job) for job in jobs]
    # hashes_grouped_by_job =  [ [h1, h2, h3], [h1, h2, h3], etc ]

    tasks_all_chained = list(itertools.chain(*tasks_grouped_by_job))
    #chain(*[[1,2], [6,7,8]]) = [1,2,6,7,8]

    hashes_all_chained = list(itertools.chain(*hashes_grouped_by_job))

    for task, hash_ in zip(tasks_all_chained, hashes_all_chained):
        tasks_queue.put((task, hash_))

    jobswriter = CompletedJobsWriter(TERM, (2,7)) #TODO: unhardcode
    for i in range(len(tasks_all_chained)):
        jobswriter.print_line(done_queue.get())

    for i in range(ARGS.nr_workers):
        tasks_queue.put('STOP')


    # Merge the sharded HDF5 files
    for i, job in enumerate(jobs):

        h = h5py.File(filename_from_job(job), 'w')

        for key in cfg[job]:
            h.attrs[key] = cfg[job][key]

        for index, hash_ in enumerate(hashes_grouped_by_job[i]):
            f = h5py.File(TEMPDIR+hash_+'.hdf5', 'r')

            h5path ="sim_"+str(index).zfill(4)+"/"
            h.create_group(h5path)
            for key in f.keys():
                h[h5path][key] = f[key].value

            f.close()

        h.close()

    logging.info('END LOG FILE : ' + time.strftime("%c"))



