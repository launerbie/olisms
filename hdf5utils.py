#!/usr/bin/env python
# name: hdf5utils.py

import os
import unittest
import h5py
import numpy 

class Dataset(object):
    def __init__(self, dset):
        """
        
        Parameters
        ----------
        dset: h5py Dataset

        """
        self.dset = dset
        self.chunkcounter = 0  
        self.blockcounter = 0
        self.chunksize = dset.chunks[0]
        self.blocksize = dset.shape[0] 
        self.arr_shape = dset.shape[1:] 

        self.dbuffer = list()

    def append(self, array):
        """
        Parameters
        ----------
        array: ndarray or list

        """
        self.dbuffer.append(array)

        if len(self.dbuffer) == self.chunksize: # WRITE BUFFER
            begin = self.blockcounter*self.blocksize + self.chunkcounter*self.chunksize
            end = begin + self.chunksize
            dbuffer_ndarray = numpy.array(self.dbuffer)
            self.dset[begin:end, ...] = dbuffer_ndarray
            self.dbuffer = list() 

            if end == self.dset.shape[0]: #BLOCK IS FULL --> CREATE NEW BLOCK
                new_shape = sum(((end+self.blocksize,), self.arr_shape), ())
                self.dset.resize(new_shape)
                self.blockcounter += 1
                self.chunkcounter = 0 
            else:
                self.chunkcounter += 1

    def flush(self, trim=True): 
        dbuffer = self.dbuffer 

        dbuffer_ndarray = numpy.array(dbuffer)

        begin = self.blockcounter*self.blocksize + self.chunkcounter*self.chunksize
        end = begin + len(dbuffer)
        self.dset[begin:end, ...] = dbuffer_ndarray
        self.dbuffer = list() 
        
        if trim:
            new_shape = sum(((end,), self.arr_shape), ())
            self.dset.resize(new_shape)


class HDF5Handler(object):

    def __init__(self, filename, mode='a'):
        """

        Parameters
        ----------
        filename   : filename of the hdf5 file.


        Usage should roughly be like:
        -----------------------------

            with HDF5Handler('test.hdf5') as h:
                while condition: #
                    h.append(ndarray, '/grp0/position')
                    h.append(ndarray, '/grp0/velocity')
                    h.append(ndarray, '/grp1/position')
                    h.append(ndarray, '/grp1/velocity')
        
        """
        self.filename = filename
        self.mode = mode
        self.index = dict()

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode) 
        return self

    def __exit__(self, extype, exvalue, traceback):
        self.flushbuffers() 
        self.file.close()
        return False

    def append(self, array, dset_path, **kwargs):
        """ 
        Parameters
        ----------
        array : ndarray or list 
        dset_path  : unix-style path ( 'group/datasetname' )

        """
        if dset_path in self.index:
            self.index[dset_path].append(array)
        else:
            self.create_dset(dset_path, array, **kwargs) 
            self.index[dset_path].append(array)

    def create_dset(self, dset_path, array, chunksize=1000, blockfactor=100):
        """
        Define h5py dataset parameters here. 


        Parameters
        ----------
        dset_path: unix-style path
        array: array to append
        blockfactor: used to calculate blocksize. (blocksize = blockfactor*chunksize)
        chunksize: determines the buffersize. (e.g.: if chunksize = 1000, the 
        buffer will be written to the dataset after a 1000 HDF5Handler.append() calls. 
        You want to  make sure that the buffersize is between 10KiB - 1 MiB = 1048576 bytes.

        This has serious performance implications if chosen too big or small,
        so I'll repeat that:
             
           MAKE SURE YOU CHOOSE YOUR CHUNKSIZE SUCH THAT THE BUFFER 
           DOES NOT EXCEED 1048576 bytes.

        See h5py docs on chunked storage for more info.

        """
        if isinstance(array, numpy.ndarray):
            arr_shape = array.shape 
        elif isinstance(array, list):
            ndarray = numpy.array(list)
            arr_shape = len(ndarray) 
        else:
            raise TypeError("{} not supported".format(type(array)))

        blocksize = blockfactor * chunksize 

        chunkshape = sum(((chunksize,), arr_shape), ())
        maxshape = sum(((None,), arr_shape), ())

        dsetkw = dict(chunks=chunkshape, maxshape=maxshape)
                                       
        init_shape = sum(((blocksize,), arr_shape), ())
        dset = self.file.create_dataset(dset_path, shape=init_shape, **dsetkw)
        self.index.update({dset_path: Dataset(dset)})

    def flushbuffers(self):
        """
        When the number of h.append calls is not a multiple of buffersize, then there 
        will be unwritten arrays in dbuffer, since dbuffer is only written when it is full.
        """
        for dset in self.index.values():
            dset.flush()
            

#TODO: move tests
class test_HDF5Handler_ndarrays_resizable(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'test.hdf5'
        self.ints = numpy.ones(123456*4).reshape(123456, 4)
        self.floats = numpy.linspace(0, 4123, 10000*3).reshape(10000, 3)

        self.kwargs = dict(chunksize=1000, blockfactor=100) #choose wisely!

    def test_group_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup'], h5py._hl.group.Group) )

    def test_hdf5file_dataset_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))
            
    def test_group_and_dataset_creation(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset', **self.kwargs)
            self.assertTrue( isinstance(h.file['testgroup/testset'], h5py._hl.dataset.Dataset) )
            self.assertTrue( isinstance(h.file['testgroup']['testset'], h5py._hl.dataset.Dataset) )

    def test_group_creation_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'testgroup/testset', **self.kwargs)

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup'], h5py._hl.group.Group) )
        
    def test_hdf5file_dataset_creation_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 
            self.assertTrue(isinstance(h.file['test'], h5py._hl.dataset.Dataset))

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['test'], h5py._hl.dataset.Dataset) )
            
    def test_group_and_dataset_creation_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row,'testgroup/testset', **self.kwargs)

        f = h5py.File(self.filename)
        self.assertTrue( isinstance(f['testgroup/testset'], h5py._hl.dataset.Dataset) )
        self.assertTrue( isinstance(f['testgroup']['testset'], h5py._hl.dataset.Dataset) )

    def test_multiple_datasets(self):
        ndarrA =  numpy.ones(10000*3).reshape(10000, 3)

        with HDF5Handler(self.filename) as h:
            for i in range(10000):
                h.append(ndarrA[i], 'testA', **self.kwargs) 
                h.append(ndarrA[i], 'testB', **self.kwargs) 
                h.append(ndarrA[i], 'testC', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarrA), f['testA'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testB'].value.sum())
        self.assertEqual(numpy.sum(ndarrA), f['testC'].value.sum())
        self.assertEqual(3, len(f.keys()) )
   
    def test_flushbuffers(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with HDF5Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
 
    def test_trimming(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with HDF5Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(ndarr.shape, f['test'].shape)

    def test_flushbuffers_and_trim(self):
        ndarr = numpy.ones(12345*3).reshape(12345, 3)

        with HDF5Handler(self.filename) as h:
            for row in ndarr:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(ndarr), f['test'].value.sum())
        self.assertEqual(ndarr.shape, f['test'].shape)

    #####################   Value tests  ####################

    def test_sum_ints_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.ints:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertEqual(numpy.sum(self.ints), f['test'].value.sum())

    def test_sum_flts_almostequal6_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(numpy.sum(self.floats), f['test'].value.sum(), places=6)

    def test_sum_flts_almostequal5_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(numpy.sum(self.floats), f['test'].value.sum(), places=5)

    def test_sum_flts_almostequal4_after_closing(self):
        with HDF5Handler(self.filename) as h:
            for row in self.floats:
                h.append(row, 'test', **self.kwargs) 

        f = h5py.File(self.filename)
        self.assertAlmostEqual(numpy.sum(self.floats), f['test'].value.sum(), places=4)

        
    def tearDown(self):
        try:
            os.remove(self.filename)
        except OSError:
            pass


if __name__ == "__main__":
    from colored import ColoredTextTestRunner
    unittest.main(verbosity=2, testRunner=ColoredTextTestRunner)



