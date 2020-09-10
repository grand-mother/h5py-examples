#! /usr/bin/env python3

import h5py
import numpy

# The following example illustrates how annotations can be added to an HDF5
# dataset. Let use a dummy array of positions for this prurpose
positions = numpy.eye(3, dtype='f8')


with h5py.File('example.hdf5', 'w') as f:
    dataset = f.create_dataset('positions', data=positions)

    # The dataset can be annotated using the attrs field, e.g. as:
    dataset.attrs['columns'] = ('x', 'y', 'z')
    dataset.attrs['units'] = 'm'


with h5py.File('example.hdf5', 'r') as f:
    dataset = f['positions'] # <== This is only a handle. It does not read
                             #     the data from file

    positions = dataset[:]   # <== Accessing / slicing the data forces reading
                             #     them from file

    # Below we read back the annotations using the attrs field, as previously:
    units = dataset.attrs['units']
    columns = dataset.attrs['columns']

print(positions)
print(units)
print(columns)
