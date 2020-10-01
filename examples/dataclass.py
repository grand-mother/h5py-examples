#! /usr/bin/env python3

from dataclasses import dataclass
import h5py
import numpy
from typing import Union


# This example illustrates how dataclasses could be serialized to HDF5 files.
# First let us define HDF5 I/O extensions as class

class H5IO:
    '''Add HDF5 support to a dataclass
    '''

    @classmethod
    def load(cls, group: Union[h5py.File, h5py.Group], lazy=False):
        '''Create a dataclass instance from a group of datasets

           By default the data are loaded from disk. Set lazy to True in order
           to postpone data access on demand. Note however that the file object
           needs to stay open in this case
        '''

        fields = {}
        for field in cls.__dataclass_fields__.keys():
            data = group[field] # <== This is only a handle. It does not read
                                #     the data from file
            if not lazy:
                data = data[:]  # <== Accessing / slicing the data forces
                                #     reading them from file
            fields[field] = data

        return cls(**fields)

    def dump(self, group: Union[h5py.File, h5py.Group]):
        '''Dump a dataclass to a group
        '''
        for field in self.__dataclass_fields__.keys():
            data = getattr(self, field)
            try:
                group[field] = data
            except KeyError:
                group.create_dataset(attr, data=data)


# Let us use an array of positions for this illustration
@dataclass
class Positions(H5IO):
    x: Union[numpy.array, h5py.Dataset, None]
    y: Union[numpy.array, h5py.Dataset, None]
    z: Union[numpy.array, h5py.Dataset, None]

positions = Positions(x=(1, 2, 3), y=(2, 3, 1), z=(3, 1, 2))


# Create an HDF5 file and dump the dataclass instance
with h5py.File('example.hdf5', 'w') as f:
    group = f.create_group('positions')
    positions.dump(group)


# Read back the dataclass instance
with h5py.File('example.hdf5', 'r') as f:
    positions = Positions.load(f['positions'])

print(positions.x)


# Read back the dataclass instance using lazy loading
with h5py.File('example.hdf5', 'r') as f:
    positions = Positions.load(f['positions'], # <== This does not read the
                               lazy=True)      #     data from file. It only
                                               #     maps the datasets to the
                                               #     dataclass fields

    print(positions.y[0]) # <== This actually reads the data. Note that the
                          #     file must be open
