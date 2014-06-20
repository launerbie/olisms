#!/usr/bin/env python

import argparse
import h5py

def main():
    f = h5py.File(args.filename, 'r')
    f.visititems(printdset)

def printdset(name, obj):
    if isinstance(obj, h5py.Dataset):
        if "unit" in obj.attrs:
            unit = eval(obj.attrs['unit'], core.__dict__)
            print(name, obj.shape, unit)
        else:
            print(name, obj.shape)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', 
                        help="hdf5 file created by sim_veras_multiplanet.py")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_arguments()
    main()

