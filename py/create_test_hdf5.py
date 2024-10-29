# hello world
import h5py
import numpy as np

# This file create a 1000x128 dataset in HDF5 format
if __name__ == "__main__":
    # create a new file
    f = h5py.File("test.hdf5", "w")

    # create a dataset
    dset = f.create_dataset("test", (1000, 128), dtype="f4")

    # write data to the dataset
    dset[...] = np.random.rand(1000, 128)

    # close the file
    f.close()
