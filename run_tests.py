#!/usr/bin/env python

import ising_model
import numpy
import unittest

class test_calc_energy_periodic(unittest.TestCase):
    def setUp(self):
        self.ising = ising_model.Ising()
        self.ising.rij = 3 
        self.ising.kolom = 3 

    def test_calculation_total_energy(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -18)


    def test_calculation_total_energy_00(self):
        I = self.ising
        model = numpy.array([[-1,1,1],
                             [1,1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_01(self):
        I = self.ising
        model = numpy.array([[1,-1,1],
                             [1,1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_02(self):
        I = self.ising
        model = numpy.array([[1,1,-1],
                             [1,1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_10(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [-1,1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_11(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,-1,1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_12(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,-1],
                             [1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_20(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [-1,1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_21(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,-1,1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

    def test_calculation_total_energy_22(self):
        I = self.ising
        model = numpy.array([[1,1,1],
                             [1,1,1],
                             [1,1,-1]])
        I.grid = model
        self.assertEqual(I.calc_energy(), -10)

class test_delta_energy_periodic_1(unittest.TestCase):
    def setUp(self):
        self.model = numpy.array([[1,1,1],
                                  [1,1,1],
                                  [1,1,1]])

        self.ising = ising_model.Ising()
        self.ising.grid = self.model
        self.ising.rij = 3 
        self.ising.kolom = 3 

    def test_flip_00(self):
        I = self.ising
        self.assertEqual(I.delta_energy((0, 0)), -8)

    def test_flip_01(self):
        I = self.ising
        self.assertEqual(I.delta_energy((0, 1)), -8)

    def test_flip_02(self):
        I = self.ising
        self.assertEqual(I.delta_energy((0, 2)), -8)

    def test_flip_10(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 0)), -8)

    def test_flip_11(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 1)), -8)

    def test_flip_12(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 2)), -8)

    def test_flip_20(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 0)), -8)

    def test_flip_21(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 1)), -8)

    def test_flip_22(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 2)), -8)



if __name__ == '__main__':
    unittest.main(verbosity=2)

       
        
    
