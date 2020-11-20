import h5py
import os
import sys
import numpy as np
import GRANDhdf5Utilities as ghdf5

# File has sections:
# SimShower  : RecShower      #here the "trace" level, we be used by longitudinal and lateral tables
# SimEfield  : RecEfield
# SimSignal  : RawData        #this part might need more thinking for treating electronics simulations, currenlty unavailable

# Each File Section has:
# one RunLevel RunInfo table with the information that is unchanged during the run. This is an instance of the corresponding data clas
# one RunLevel RunIndex table with the information that is going to be used for quick indexing of the Run Events, usefull for searchs and filtering and maybe high level analysis. This is an array.
# one RunLevel DetectorInfo table with the information from the antennas that is constant in all the run
# one data group per event. Each event will have
#     one EventInfo table with the information for the event that changes event by event (if the data is constant over all events , that info should be in the RunInfo unless it really usefull to replicate it on every event!)
#     one DetectorInfo table with the information per antenna that changes event by event (if the data is constant over all events , that info should be in the DetectorInfo unless it really usefull to replicate it on every event!)
#     one group per detector
#        one trace per channel
#
# since tables will have fields that might be empty, we always start by creating an empty instance, and then write the information we know

#https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
#print(type(SimEfield_DetectorInfo_data))
#print(type(SimEfield_DetectorInfo_data['det_id'])) #this gives me access to the column (a numpy array)
#print(type(SimEfield_DetectorInfo_data['det_id'][:])) #this gives me the contents of the column (a numpy array)
#print(type(SimEfield_DetectorInfo_data['det_id'][0])) #and this gives me the first element (of the content, not access to the file)
#print(type(SimEfield_DetectorInfo_data[0,'det_id'])) #so, this gives me access to the 'det_id' field of record 0
#print(SimEfield_DetectorInfo_data[0])


def AppendRowToDataset(dataset, newrow):
    #
    #check if dataset has at least one
    #
    shape=list(dataset.shape) #get the shape into a list
    shape[0]=shape[0]+1 #add 1 to the first dimensin
    shape=tuple(shape) #put it back into a tuple
    dataset.resize(shape)
    dataset[shape[0]-1]=newrow
