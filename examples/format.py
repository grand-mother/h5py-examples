#! /usr/bin/env python3

import h5py
import numpy

# The following example shows how h5py interoperates with numpy arrays.
# Let us use a dummy numpy array for this purpose. Let us store data using
# double precision
data = numpy.zeros((10, 3), 'f8') # <== This maps a double[10][3] C array in
                                  #     memory


with h5py.File('example.hdf5', 'w') as f:
    # By default the shape and data type of the HDF5 dataset are inferred from
    # the NumPy array (shape, dtype attributes). E.g.
    dataset = f.create_dataset('data', data=data)

    # We can override the shape and data type when storing the data to file,
    # E.g. as:
    dataset = f.create_dataset('data_flat_float', data=data,
                               shape=(data.size,), # <== Here we store the data
                                                   #     flat as a 1d array

                               dtype='f4') # <== Here we store the data in file
                                           #     using float instead of double


with h5py.File('example.hdf5', 'r') as f:
    dataset = f['data_flat_float'] # <== This is a handle. It does not load the
                                   #     data from file. Data are loaded when
                                   #     accessed

    flat_data = numpy.array(dataset,
                            dtype='f8') # <== This forces loading the data.
                                        #     In addition we store them
                                        #     back in memory using double
                                        #     precision

    data = flat_data.reshape((10, 3)) # <== This restores the initial shape.
                                      #     Note that is does not copy the data!
                                      #     Only the way there are accessed is
                                      #     changed. I.e. flat_data and data
                                      #     refer to the same memory

print(data)
flat_data[3] = 1
print(data[1,0])
