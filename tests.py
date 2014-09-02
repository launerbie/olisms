#!/usr/bin/env python
import argparse
import numpy
import unittest
import ising
from ext.colored import ColoredTextTestRunner

from ext.hdf5handler.tests import test_file_group_dataset_creation
from ext.hdf5handler.tests import test_python_scalars
from ext.hdf5handler.tests import test_python_lists
from ext.hdf5handler.tests import test_python_tuples
from ext.hdf5handler.tests import test_ndarrays
from ext.hdf5handler.tests import test_prefix


class test_TotalEnergy_2D(unittest.TestCase):
    def setUp(self):
        self.ising = ising.Ising(shape=(3,3), sweeps=1)

    def test_calculation_total_energy(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -18)


    def test_calculation_total_energy_00(self):
        I = self.ising
        model = numpy.array([[-1,1,1],
                             [1,1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_01(self):
        I = self.ising
        model = numpy.array([[1,-1,1],
                             [1,1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_02(self):
        I = self.ising
        model = numpy.array([[1,1,-1],
                             [1,1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_10(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [-1,1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_11(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,-1,1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_12(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,-1],
                             [1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_20(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [-1,1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_21(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,-1,1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_22(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,1,-1]])
        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -10)


class test_delta_energy_2D(unittest.TestCase):
    def setUp(self):
        self.model = numpy.array([[1,1,1],
                                  [1,1,1],
                                  [1,1,1]])

        self.ising = ising.Ising(shape=[3,3], sweeps=10)
        self.ising.set_grid(self.model)

    def test_flip_00(self):
        I = self.ising
        self.assertEqual(I.delta_energy(0), 8)

    def test_flip_01(self):
        I = self.ising
        self.assertEqual(I.delta_energy(1), 8)

    def test_flip_02(self):
        I = self.ising
        self.assertEqual(I.delta_energy(2), 8)

    def test_flip_10(self):
        I = self.ising
        self.assertEqual(I.delta_energy(3), 8)

    def test_flip_11(self):
        I = self.ising
        self.assertEqual(I.delta_energy(4), 8)

    def test_flip_12(self):
        I = self.ising
        self.assertEqual(I.delta_energy(5), 8)

    def test_flip_20(self):
        I = self.ising
        self.assertEqual(I.delta_energy(6), 8)

    def test_flip_21(self):
        I = self.ising
        self.assertEqual(I.delta_energy(7), 8)

    def test_flip_22(self):
        I = self.ising
        self.assertEqual(I.delta_energy(8), 8)



########################################################################
######################### TEST 3D ISING ################################
########################################################################

class test_TotalEnergy_3D(unittest.TestCase):
    def setUp(self):
        self.ising = ising.Ising(shape=[3,3,3], sweeps=1)

    def test_calculation_total_energy(self):
        I = self.ising
        model = numpy.ones((3,3,3))

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -81)

###################### z = 0 ##########################
    def test_calculation_total_energy_000(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,0,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_100(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,0,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_200(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,0,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_010(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,1,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_110(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,1,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_210(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,1,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_020(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,2,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_120(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,2,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_220(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,2,0] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

###################### z = 1 ##########################
    def test_calculation_total_energy_001(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,0,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_101(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,0,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_201(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,0,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_011(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,1,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_111(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,1,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_211(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,1,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_021(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,2,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_121(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,2,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_221(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,2,1] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

###################### z = 2 ##########################
    def test_calculation_total_energy_002(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,0,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_102(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,0,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_202(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,0,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_012(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,1,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_112(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,1,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_212(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,1,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_022(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[0,2,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_122(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[1,2,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)

    def test_calculation_total_energy_222(self):
        I = self.ising
        model = numpy.ones((3,3,3))
        model[2,2,2] = -1

        I.set_grid(model)
        self.assertEqual(I.calc_energy(), -69)


class test_delta_energy_3D(unittest.TestCase):
    def setUp(self):
        self.model = numpy.ones((3,3,3))
        self.ising = ising.Ising(shape=[3,3,3], sweeps=1)
        self.ising.set_grid(self.model)

###################### z = 0 ##########################
    def test_flip_000(self):
        I = self.ising
        self.assertEqual(I.delta_energy(0), 12)

    def test_flip_100(self):
        I = self.ising
        self.assertEqual(I.delta_energy(1), 12)

    def test_flip_200(self):
        I = self.ising
        self.assertEqual(I.delta_energy(2), 12)

    def test_flip_010(self):
        I = self.ising
        self.assertEqual(I.delta_energy(3), 12)

    def test_flip_020(self):
        I = self.ising
        self.assertEqual(I.delta_energy(4), 12)

    def test_flip_110(self):
        I = self.ising
        self.assertEqual(I.delta_energy(5), 12)

    def test_flip_120(self):
        I = self.ising
        self.assertEqual(I.delta_energy(6), 12)

    def test_flip_210(self):
        I = self.ising
        self.assertEqual(I.delta_energy(7), 12)

    def test_flip_220(self):
        I = self.ising
        self.assertEqual(I.delta_energy(8), 12)

###################### z = 1 ##########################
    def test_flip_001(self):
        I = self.ising
        self.assertEqual(I.delta_energy(9), 12)

    def test_flip_101(self):
        I = self.ising
        self.assertEqual(I.delta_energy(10), 12)

    def test_flip_201(self):
        I = self.ising
        self.assertEqual(I.delta_energy(11), 12)

    def test_flip_011(self):
        I = self.ising
        self.assertEqual(I.delta_energy(12), 12)

    def test_flip_021(self):
        I = self.ising
        self.assertEqual(I.delta_energy(13), 12)

    def test_flip_111(self):
        I = self.ising
        self.assertEqual(I.delta_energy(14), 12)

    def test_flip_121(self):
        I = self.ising
        self.assertEqual(I.delta_energy(15), 12)

    def test_flip_211(self):
        I = self.ising
        self.assertEqual(I.delta_energy(16), 12)

    def test_flip_221(self):
        I = self.ising
        self.assertEqual(I.delta_energy(17), 12)

###################### z = 2 ##########################
    def test_flip_002(self):
        I = self.ising
        self.assertEqual(I.delta_energy(18), 12)

    def test_flip_102(self):
        I = self.ising
        self.assertEqual(I.delta_energy(19), 12)

    def test_flip_202(self):
        I = self.ising
        self.assertEqual(I.delta_energy(20), 12)

    def test_flip_012(self):
        I = self.ising
        self.assertEqual(I.delta_energy(21), 12)

    def test_flip_022(self):
        I = self.ising
        self.assertEqual(I.delta_energy(22), 12)

    def test_flip_112(self):
        I = self.ising
        self.assertEqual(I.delta_energy(23), 12)

    def test_flip_122(self):
        I = self.ising
        self.assertEqual(I.delta_energy(24), 12)

    def test_flip_212(self):
        I = self.ising
        self.assertEqual(I.delta_energy(25), 12)

    def test_flip_222(self):
        I = self.ising
        self.assertEqual(I.delta_energy(26), 12)

########################################################################
######################### END OF 3D ISING TEST #########################
########################################################################


class test_magnetization(unittest.TestCase):
    def setUp(self):
        self.model = numpy.array([[-1,-1,-1],
                                  [1,1,1],
                                  [1,1,1]])

        self.ising = ising.Ising(shape=[3,3], sweeps=1)
        self.ising.set_grid(self.model)

    def test_magnetization(self):
        I = self.ising
        self.assertEqual(I.magnetization, 3)

#if __name__ == '__main__':
#    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--verbosity', type=int, default=2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()

    test_cases = [\
                  test_TotalEnergy_2D,
                  test_TotalEnergy_3D,
                  test_delta_energy_2D,
                  test_delta_energy_3D,
                  test_magnetization,
                  # TODO: import HDF5 stuff as a suite
                  test_file_group_dataset_creation,
                  test_python_scalars,
                  test_python_lists,
                  test_ndarrays,
                  test_prefix,
                 ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for tc in test_cases:
        tests = loader.loadTestsFromTestCase(tc)
        suite.addTests(tests)

    runner = ColoredTextTestRunner(verbosity=args.verbosity)
    results = runner.run(suite)

    if (len(results.failures) or len(results.errors)) > 0:
        exit(1)
    else:
        exit(0)


