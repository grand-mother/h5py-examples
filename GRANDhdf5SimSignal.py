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
# one RunLevel EventIndex table with the information that is going to be used for quick indexing of the Run Events, usefull for searchs and filtering and maybe high level analysis. This is an array.
# one RunLevel DetectorIndex table with the information from the antennas that is constant in all the run
# one data group per event. Each event will have
#     one EventInfo table with the information for the event that changes event by event (if the data is constant over all events , that info should be in the RunInfo unless it really usefull to replicate it on every event!)
#     one DetectorIndex table with the information per antenna that changes event by event (if the data is constant over all events , that info should be in the DetectorIndex unless it really usefull to replicate it on every event!)
#     one group per detector
#        one trace per channel
#
# since tables will have fields that might be empty, we always start by creating an empty instance, and then write the information we know

#https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
#print(type(SimSignal_DetectorIndex_data))
#print(type(SimSignal_DetectorIndex_data['det_id'])) #this gives me access to the column (a numpy array)
#print(type(SimSignal_DetectorIndex_data['det_id'][:])) #this gives me the contents of the column (a numpy array)
#print(type(SimSignal_DetectorIndex_data['det_id'][0])) #and this gives me the first element (of the content, not access to the file)
#print(type(SimSignal_DetectorIndex_data[0,'det_id'])) #so, this gives me access to the 'det_id' field of record 0
#print(SimSignal_DetectorIndex_data[0])

#SimSignal RunLevelInfo
#prefix=se_rifo
SimSignal_RunInfo_dtype =np.dtype  ([('run_id', 'uint32'),            #RunID: Just to be sure we are in the right place
                                     ('signal_sim','S20'),
                                     ('filter' ,'S16'),           #Name of the filter
                                     ('filter_param','f8',(1,2)), #parameters of the filter (for now, filter has only low and high limits, in MHz
                                     #First Level Parameters
                                     #Second Level Parameters
                                     ('trig_thresh','f4')
                                    ])

#SimSignal RunLevelIndex
#prefix=se_ri
SimSignal_EventIndex_dtype =np.dtype  ([('evt_id', 'S40'),    #EventID: Just to be sure we are in the right place
                                      ('evt_name','S100'), #ZHAireS TaskName usefull to keep to find the original files
                                     #Second Level Parameters
                                      ('n_trig','i')       #Number of triggered antennas
                                     ])


#SimSignal EventLevelInfo
#prefix=se_ei
SimSignal_EventInfo_dtype =np.dtype  ([('run_id', 'uint32'),      #RunID: Just to be sure we are in the right place. At some point, we might want to select events and put them together in a file...good to know where they came from
                                       ('evt_id', 'S40'),      #AntenaID:
                                       ('evt_name','S100'),   #ZHAireS TaskName
                                       #Second Level Parameters
                                       ('n_trig','i')         #Number of triggered antennas
                                      ])


#PerEventDetectorLevelInfo  (Is there a Run level antenna info? YES)
#prefix=se_di
SimSignal_DetectorIndex_dtype =np.dtype  ([('det_id', 'S20'),         #AntenaID: So that we are sure we are in the right place
                                         ('det_type','S20'),         #Antena Type:We might have different designs. Irrelevant for the electric field sim, but might be handy
                                         ('det_pos_shc','f4',(1,3)), #position in 32bit (single) precision
                                         ('t_0','f4'),
                                         #FirstLevelSignalParameters
                                         ('p2p','f4',(1,3)),         #Peak to Peak amplitudes in each channel
                                         #SecondLevelSingalParmeters
                                         ('trig','i',(1,3))          #Flag the channels that triggerd (what was the trigger condition?)
                                        ])

#Signal Trace
#We dont need a special datatype for traces for now. This solves the problem of the variable lenght (but we will have to store the T0, tmin, tmax, tbinsize)


#Run Level
def SimSignalAddRunInfo(filehandle, RunID,SimSignal_RunInfo=None ):

    node="Run_"+str(RunID)+"/SimSignal_RunInfo"

    SimSignal_RunInfo_data= ghdf5.AddToInfo(filehandle, node, SimSignal_RunInfo_dtype,SimSignal_RunInfo )

    return SimSignal_RunInfo_data

def SimSignalAddEventIndex(filehandle, RunID,SimSignal_EventIndex=None):

    node="Run_"+str(RunID)+"/SimSignal_EventIndex"

    SimSignal_EventIndex_data, item_index = ghdf5.AddToIndex(filehandle, node, SimSignal_EventIndex_dtype, 'evt_id', SimSignal_EventIndex)

    return SimSignal_EventIndex_data, item_index

#this is the function that compiles the information of the event that will go to the EventIndex
def SimSignalCompileEventIndex(filehandle, RunID, EventID):

    #Information that we will extract from the SimSignal_EventInfo
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimSignal_EventInfo" already exist. If it does, give an error (or handle overwriting with an optional parameter).
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimSignal_EventInfo"
    exists= node in filehandle
    if(exists):
      #grab it so we can read the data
      SimSignal_EventInfo=filehandle[node]
      #create empty instance for EventIndex table
      SimSignal_EventIndex= np.zeros(1,SimSignal_EventIndex_dtype)
      #fill with the values i want
      SimSignal_EventIndex['evt_id']=SimSignal_EventInfo['evt_id']
      SimSignal_EventIndex['evt_name']=SimSignal_EventInfo['evt_name']
      SimSignal_EventIndex['n_trig']=SimSignal_EventInfo['n_trig']
      return SimSignal_EventIndex
    else:
        print("SimSignalCompileEventIndex:Could not find ",node)
        return None


#Event Level
def SimSignalAddEventInfo(filehandle, RunID, EventID, SimSignal_EventInfo=None):
    #
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimSignal_EventInfo"
    SimSignal_EventInfo_data= ghdf5.AddToInfo(filehandle, node, SimSignal_EventInfo_dtype,SimSignal_EventInfo )

    return SimSignal_EventInfo_data

def SimSignalAddDetectorIndex(filehandle, RunID, EventID, SimSignal_DetectorIndex=None):
    #
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimSignal_DetectorIndex"

    SimSignal_DetectorIndex_data, item_index = ghdf5.AddToIndex(filehandle, node, SimSignal_DetectorIndex_dtype, 'det_id', SimSignal_DetectorIndex)

    return SimSignal_DetectorIndex_data, item_index


#trace level
#we might want to wrap up everything in a function that saves the trace, and updates/creates the DetectorIndex table
def SimSignalWriteSimSignal(filehandle, RunID, EventID, DetectorID,TraceX,TraceY,TraceZ):
    #For the Trace, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimSignal_X" exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    nodeX="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimSignal_X"
    existsX = nodeX in filehandle
    nodeY="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimSignal_Y"
    existsY = nodeY in filehandle
    nodeZ="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimSignal_Z"
    existsZ = nodeZ in filehandle
    if(existsX or existsY or existsZ):
      print("SimSignalWriteSimSignal: Event exist, not updated",RunID,EventID,DetectorID)
      return 0

    SimSignal_X_data=filehandle.create_dataset(nodeX, data=TraceX, dtype='f4')
    SimSignal_Y_data=filehandle.create_dataset(nodeY, data=TraceY, dtype='f4')
    SimSignal_Z_data=filehandle.create_dataset(nodeZ, data=TraceZ, dtype='f4')
    #return SimSignal_X_data,SimSignal_Y_data,SimSignal_Z_data


