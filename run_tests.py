#!/usr/bin/env python
import argparse
import unittest
import sys
import os
import importlib

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__" and __package__ is None:
    ARGS = get_arguments()

    # __file__ = "run_tests.py"
    SCRIPT_PATH = os.path.realpath(__file__)   # "/some/path/to/mypackage/run_tests.py"
    PACKAGE_DIR = os.path.dirname(SCRIPT_PATH) # "/some/path/to/mypackage"
    PARENT_PACKAGE_DIR, PACKAGE_NAME = os.path.split(PACKAGE_DIR) #("/some/path/to", "mypackage")

    sys.path.append(PARENT_PACKAGE_DIR)

    olisms = importlib.import_module(PACKAGE_NAME)
    __package__ = PACKAGE_NAME

    from olisms.ext.colored import ColoredTextTestRunner
    from olisms.tests import tests as isingstests
    from olisms.ext.hdf5handler.tests import tests as hdf5handlertests

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(isingstests))
    suite.addTests(loader.loadTestsFromModule(hdf5handlertests))

    runner = ColoredTextTestRunner(verbosity=ARGS.verbosity)
    results = runner.run(suite)

    if (len(results.failures) or len(results.errors)) > 0:
        exit(1)
    else:
        exit(0)

