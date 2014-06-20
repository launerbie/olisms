#!/usr/bin/env python

import os
import h5py
import numpy
import unittest

import ising_model
from hdf5utils import HDF5Handler

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
        self.assertEqual(I.delta_energy((0, 0)), 8)

    def test_flip_01(self):
        I = self.ising
        self.assertEqual(I.delta_energy((0, 1)), 8)

    def test_flip_02(self):
        I = self.ising
        self.assertEqual(I.delta_energy((0, 2)), 8)

    def test_flip_10(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 0)), 8)

    def test_flip_11(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 1)), 8)

    def test_flip_12(self):
        I = self.ising
        self.assertEqual(I.delta_energy((1, 2)), 8)

    def test_flip_20(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 0)), 8)

    def test_flip_21(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 1)), 8)

    def test_flip_22(self):
        I = self.ising
        self.assertEqual(I.delta_energy((2, 2)), 8)


class test_magnetization(unittest.TestCase):
    def setUp(self):
        self.model = numpy.array([[-1,-1,-1],
                                  [1,1,1],
                                  [1,1,1]])

        self.ising = ising_model.Ising()
        self.ising.grid = self.model
        self.ising.rij = 3 
        self.ising.kolom = 3 

    def test_magnetization(self):
        I = self.ising
        self.assertEqual(I.magnetization, 3)


class test_HDF5Handler_ndarrays(unittest.TestCase):
    def setUp(self):
        self.Handler = HDF5Handler
        self.filename = 'test.hdf5'

        self.ints1d = numpy.ones(12345*2)
        self.floats1d = numpy.linspace(0, 4123, 10000*2)
        self.ints = numpy.ones(12345*2).reshape(12345, 2)
        self.floats = numpy.linspace(0, 4123, 10000*2).reshape(10000, 2)

        self.sumints1d = numpy.sum(self.ints1d)
        self.sumfloats1d = numpy.sum(self.floats1d)
        self.sumints = numpy.sum(self.ints)
        self.sumfloats = numpy.sum(self.floats)

        #TODO: write a benchmark module to test different shapes
        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!

    def test_group_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset')
            self.assertTrue( isinstance(h.file['testgroup'], h5py.Group) )

    def test_hdf5file_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 
            self.assertTrue(isinstance(h.file['test'], h5py.Dataset))
            
    def test_group_and_dataset_creation(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset')
            self.assertTrue( isinstance(h.file['testgroup/testset'], h5py.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['testset'], h5py.Dataset) )

    def test_group_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset')

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup'], h5py.Group) )
        
    def test_hdf5file_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 
            self.assertTrue(isinstance(h.file['test'], h5py.Dataset))

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['test'], h5py.Dataset) )
            
    def test_group_and_dataset_creation_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset')

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup/testset'], h5py.Dataset) )
        self.assertTrue( isinstance(f['testgroup']['testset'], h5py.Dataset) )

    def test_creation_multiple_datasets(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testA') 
                h.append(row, 'testB') 
                h.append(row, 'testC') 
            self.assertTrue(isinstance(h.file['testA'], h5py.Dataset) )
            self.assertTrue(isinstance(h.file['testB'], h5py.Dataset) )
            self.assertTrue(isinstance(h.file['testC'], h5py.Dataset) )
            self.assertEqual(3, len(h.file.keys()) )

    def test_creation_multiple_datasets_after_closing(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testA') 
                h.append(row, 'testB') 
                h.append(row, 'testC') 

        f = h5py.File(self.filename)
        self.assertTrue(isinstance(f['testA'], h5py.Dataset) )
        self.assertTrue(isinstance(f['testB'], h5py.Dataset) )
        self.assertTrue(isinstance(f['testC'], h5py.Dataset) )
        self.assertEqual(3, len(f.keys()) )
   
    def test_flushbuffers(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())
 
    def test_trimming(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.ints.shape, f['test'].shape)

    def test_flushbuffers_and_trim(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())
        self.assertEqual(self.ints.shape, f['test'].shape)

    def test_shape_scalars(self):
        with self.Handler(self.filename) as h:
            for row in self.ints1d:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.ints1d.shape, f['test'].shape)

    def test_shape_arrays(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.ints.shape, f['test'].shape)
        

    #####################   Value tests  ####################

    def test_sum_ints_scalar(self):
        with self.Handler(self.filename) as h:
            for element in self.ints1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints1d, f['test'].value.sum())

    def test_sum_flts_scalar_almostequal7(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=7)

    def test_sum_flts_scalar_almostequal6(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=6)

    def test_sum_flts_scalar_almostequal5(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=5)


    def test_sum_flts_scalar_almostequal4(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=4)


    def test_sum_flts_scalar_almostequal3(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=3)


    def test_sum_flts_scalar_almostequal2(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=2)


    def test_sum_flts_scalar_almostequal1(self):
        with self.Handler(self.filename) as h:
            for element in self.floats1d:
                h.append(element, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats1d, f['test'].value.sum(), places=1)

         
    def test_sum_ints_array(self):
        with self.Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['test'].value.sum())

    def test_sum_flts_array_almostequal7(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=7)

    def test_sum_flts_array_almostequal6(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=6)

    def test_sum_flts_array_almostequal5(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=5)

    def test_sum_flts_array_almostequal4(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=4)

    def test_sum_flts_array_almostequal3(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=3)

    def test_sum_flts_array_almostequal2(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=2)

    def test_sum_flts_array_almostequal1(self):
        with self.Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(self.sumfloats, f['test'].value.sum(), places=1)

    def test_prefix(self):
        with self.Handler(self.filename) as h:
            h.prefix = 'prefix/'
            for row in self.ints:
                h.append(row, 'test') 

        f = h5py.File(self.filename)
        self.assertEqual(self.sumints, f['prefix/test'].value.sum())

        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass

if __name__ == '__main__':
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)

       
        
    
