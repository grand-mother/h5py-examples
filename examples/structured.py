#! /usr/bin/env python3

import h5py
import numpy

# Structured arrays are C arrays of structures. They can be created with NumPy
# by specifying named data types
positions = numpy.array([(1, 2, 3), (2, 3, 1), (3, 1, 2)],
                        dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')])

print(positions.shape) # <== Note that this is the number of structures i.e.
                       #     shape = (3,) not (3, 3)


# Structured arrays are automatically managed by H5py
with h5py.File('example.hdf5', 'w') as f:
    dataset = f.create_dataset('positions', data=positions)


with h5py.File('example.hdf5', 'r') as f:
    dataset = f['positions'] # <== This is only a handle. It does not read
                             #     the data from file

    positions = dataset[:]   # <== Accessing / slicing the data forces reading
                             #     them from file

print(positions['x'])
print(positions[0]['y'])

# Note that the following does not work, despite the structured array having the
# same memory layout than a double[3][3] C array:
#
# print(positions([0,0]))
