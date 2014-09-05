#!/usr/bin/env python
import argparse
import unittest
import sys
import os

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__" and __package__ is None:
    ARGS = get_arguments()
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import olisms           #TODO: determine packagename from path,
    __package__ = "olisms"  #since now this will break if top-level dir is not called 'olisms'

    from .ext.colored import ColoredTextTestRunner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    from .tests import tests as isingstests
    from .ext.hdf5handler.tests import tests as hdf5handlertests
    suite.addTests(loader.loadTestsFromModule(isingstests))
    suite.addTests(loader.loadTestsFromModule(hdf5handlertests))

    runner = ColoredTextTestRunner(verbosity=ARGS.verbosity)
    results = runner.run(suite)

    if (len(results.failures) or len(results.errors)) > 0:
        exit(1)
    else:
        exit(0)

