#! /usr/bin/env python3

# The following example illustrates how numerical data can be wrapped with
# units using the Pint package.
#
# Note that while using Physical units can prevent some type of bugs it can
# also create new ones. In particular when wrapping data with units one must
# take care if a copy or a reference of the data is returned. Performance wise
# references are better for large data sets however they can lead to unexpected
# results, i.e. bugs.

import h5py
import numpy


# First we create a new unit registry and define some shortcuts. Note that this
# would actually be imported from a dedicated package, e.g. as:
# `from grand import Quantity, units`
import pint
units = pint.UnitRegistry()
Quantity = units.Quantity

# The unit registry starts populated with standard units. Extra / custom units
# can be added using a definition file or programmatically, e.g. as:
units.define('bigfoot = 3 * meter = Bf')
# See: https://pint.readthedocs.io/en/0.10.1/defining.html#defining for more
# examples


# Let us use dummy positions data for this illustration. The Quantity
# constructor allows to wrap data with a unit. Note that the new Quantity holds
# a reference to the initial data, i.e. the data are NOT copied
data = numpy.eye(3, dtype='f8')
positions = Quantity(data, units.m)

# If a copy is desired one can use the multiplication operator instead, as:
positions_copy = data * units.m

# Changing the units inplace is done with the `ito` method.
positions.ito(units.cm) # <== This changes the unit inplace to cm. Note
                        #     that the data are also converted accordingly,
                        #     i.e. the initial data array IS modified

print(data[0,0]) # <== Now data[0,0] is 100, no more 1!

# If a copy is desired then the `to` method can be used instead, e.g. as:
positions_cm = positions.to(units.cm)
# Note however that IF the initial Quantity already has the desired unit then
# a REFERENCE is returned instead of a copy! Therefore the following code would
# be better if a copy is desired in all cases:
if positions.units == units.cm:
    positions_cm = positions.copy()
else:
    positions_cm = positions.to(units.cm)

data[0, 0] = 0
print(positions[0,0])       # <== This is 0 because it refers to data. The
print(positions_copy[0,0])  #     copies below are not modified as expected
print(positions_cm[0,0])
positions[0, 0] = 100 * units.cm


# Let write and read back the Quantity to and HDF5 file. Pint is not natively
# supported by H5Py. Therefore we use an attribute (units) in order to store
# the units information
with h5py.File('example.hdf5', 'w') as f:
    dataset = f.create_dataset('positions',
                               data=positions.magnitude) # <== Those are the
                                                         #     numerical values

    dataset.attrs['units'] = str(positions.units) # <== A string representation
                                                  #     of the unit is written
                                                  #     to the HDF5 file


with h5py.File('example.hdf5', 'r') as f:
    dataset = f['positions'] # <== This is only a handle. It does not read
                             #     the data from file

    positions = Quantity(dataset[:],              # <== Build the Quantity from
                         dataset.attrs['units'])  #     numerical data and the
                                                  #     unit string


print(positions)
print(type(positions.magnitude))
print(numpy.all(positions == positions_copy))
