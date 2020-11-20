# Examples of H5Py usage

- [examples/format.py](examples/format.py) illustrates how h5py interoperates
  with NumPy.

- [examples/structured.py](examples/structured.py) illustrates how NumPy
  structured arrays (arrays of C structures) can be created, written and read
  back.

- [examples/attributes.py](examples/attributes.py) illustrates how attributes
  can be used in order to annotate HDF5 data.

- [examples/units.py](examples/units.py) illustrates how data can be wrapped
  with units using the [Pint](https://pint.readthedocs.io/en/stable) package.

- [examples/dataclass.py](examples/dataclass.py) illustrates how dataclasses
  could be serialized to HDF5 files.

- If you want to test the ZHAireSRawToGRANDhdf5.py, uncompress the example event and do

$PYTHONINTERPRETER ZHAireSRawToGRANDHDF5.py ./example_event/ standard 0 3 TestFile.hdf5

You can run it several times with diferent run numbers and IDs to create a file with multiple times the same event.
You can get other events elswhere if you want.

You will need ZHAireS-Python AiresInfoFunctions.py from https://github.com/mjtueros/ZHAireS-Python  (from the DevelopmentLeia branch)
