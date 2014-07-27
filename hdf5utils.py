#!/usr/bin/env python
# hdf5utils.py
# -*- coding: utf-8 -*-

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
    """
    Usage should roughly be like:
    -----------------------------

        with HDF5Handler('test.hdf5') as h:
            while condition: #
                h.append(ndarray, '/grp0/position')
                h.append(ndarray, '/grp0/velocity')
                h.append(ndarray, '/grp1/position')
                h.append(ndarray, '/grp1/velocity')

    """
    def __init__(self, filename, mode='a', prefix=None):
        """
        Parameters
        ----------
        filename   : filename of the hdf5 file.
        
        """
        self.filename = filename
        self.mode = mode
        self.prefix = prefix
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
        if is_numeric(array):
            ndarray = convert_to_ndarray(array)
        else:
            raise TypeError("{} is not supported".format(type(array)))

        if self.prefix is not None:
           fulldsetpath = self.prefix+dset_path
        else:
           fulldsetpath = dset_path


        if fulldsetpath in self.index:
            self.index[fulldsetpath].append(ndarray)
        else:
            self.create_dset(fulldsetpath, array, **kwargs)
            self.index[fulldsetpath].append(ndarray)


    def create_dset(self, dset_path, array, chunksize=1000, blockfactor=100, dtype='float64'):
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
        if is_numeric(array):
            arr_shape = get_shape(array)
        else:
            raise TypeError("{} is not supported".format(type(array)))

        blocksize = blockfactor * chunksize

        chunkshape = sum(((chunksize,), arr_shape), ())
        maxshape = sum(((None,), arr_shape), ())

        dsetkw = dict(chunks=chunkshape, maxshape=maxshape, dtype=dtype)

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

def convert_to_ndarray(array):
    #TODO: this is too similar too get_shape. Rethink implementation
    if is_scalar(array):
        return array

    else: #convert tuple/list/ndarray
        if isinstance(array, numpy.ndarray):
            ndarray = array
        elif isinstance(array, (list, tuple)):
            ndarray = numpy.array(array)
        else:
            raise TypeError

    return ndarray

def get_shape(array):
    """
    returns shape of array or return (1,) if it is a scalar.
    """
    if is_scalar(array):
        arr_shape = ()
    else:
        if isinstance(array, numpy.ndarray):
            arr_shape = array.shape
        elif isinstance(array, (list, tuple)):
            #probably easiest way to determine shape of n-dimensionallists/tuples
            ndarray = numpy.array(array)
            arr_shape = ndarray.shape
        else:
            raise TypeError("shape of {} could not be determined".format(type(array)))
    return arr_shape


#TODO: kind of general stuff, maybe move elsewhere
def is_numeric(number):
    try:
        number/1.0
        return True
    except TypeError as exc:
        print(exc)
        return False

def is_scalar(array):
    try:
        len(array)
        return False
    except TypeError:
        return True





