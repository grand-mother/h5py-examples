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

#SimEfield RunLevelInfo
#prefix=se_rifo
SimEfield_RunInfo_dtype =np.dtype  ([('run_id', 'S40'),                  #RunID: Just to be sure we are in the right place
                                     ('field_sim','S20'),               #Name and version of the field simulator
                                     ('refractivity_model' ,'S16'),     #Name of the index of refraction model used ("i.e Exponential")
                                     ('refractivity_param','f8',(1,2)), #Refractivity at sea level, scale height (we will need more here!)
                                     #Second Level Parameters
                                     ('trig_thresh','f4')
                                    ])

#SimEfield RunLevelIndex
#prefix=se_ri
SimEfield_EventIndex_dtype =np.dtype  ([('evt_id', 'S40'),               #EventID: Just to be sure we are in the right place
                                      ('evt_name','S100'),              #ZHAireS TaskName usefull to keep to find the original files
                                     #Second Level Parameters
                                      ('n_trig','i')                    #Number of triggered antennas
                                     ])


#SimEfield EventLevelInfo
#prefix=se_ei
SimEfield_EventInfo_dtype =np.dtype  ([('run_id', 'uint32'),            #RunID: Just to be sure we are in the right place. At some point, we might want to select events and put them together in a file...good to know where they came from
                                       ('evt_id', 'S40'),               #EventID:
                                       ('evt_name','S100'),             #ZHAireS TaskName
                                       ('t_pre','f4'),                  #Antena Time window (ns)
                                       ('t_post','f4'),                 #Antena Time window (ns)
                                       ('t_bin_size','f4'),             #Time bin size in ns (having the number of time bins might be handy?)
                                       ('exp_xmax_pos_shc','f4',(1,3)), #Xmax is used for the timing model. It should be similar to the true one, and this is here to allow for checks
                                       #Second Level Parameters
                                       ('n_trig','i')                   #Number of triggered antennas
                                      ])


#PerEventDetectorLevelInfo  (Is there a Run level antenna info? YES)
#prefix=se_di
SimEfield_DetectorIndex_dtype =np.dtype  ([('det_id', 'S20'),           #AntenaID: So that we are sure we are in the right place
                                         ('det_type','S20'),            #Antena Type:We might have different designs. Irrelevant for the electric field sim, but might be handy
                                         ('det_pos_shc','f4',(1,3)),    #position in 32bit (single) precision
                                         ('t_0','f4'),
                                         #FirstLevelSignalParameters
                                         ('p2p','f4',(1,3)),            #Peak to Peak amplitudes in each channel
                                         #SecondLevelSingalParmeters
                                         ('trig','i',(1,3))             #Flag the channels that triggerd (what was the trigger condition?)
                                        ])

#Efield Trace
#We dont need a special datatype for traces for now. This solves the problem of the variable lenght (but we have to store the T0, tmin, tmax, tbinsize)


#Run Level
def SimEfieldAddRunInfo(filehandle, RunID, SimEfield_RunInfo=None ):

    node="Run_"+str(RunID)+"/SimEfield_RunInfo"

    SimEfield_RunInfo_data= ghdf5.AddToInfo(filehandle, node, SimEfield_RunInfo_dtype,SimEfield_RunInfo )

    return SimEfield_RunInfo_data


def SimEfieldAddEventIndex(filehandle, RunID, SimEfield_EventIndex=None):

    node="Run_"+str(RunID)+"/SimEfield_EventIndex"

    SimEfield_EventIndex_data, item_index = ghdf5.AddToIndex(filehandle, node, SimEfield_EventIndex_dtype, 'evt_id',SimEfield_EventIndex)

    return SimEfield_EventIndex_data, item_index


#this is the function that compiles the information of the event that will go to the EventIndex
def SimEfieldCompileEventIndex(filehandle, RunID, EventID):

    #Information that we will extract from the SimEfield_EventInfo
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_EventInfo" already exist. If it does, give an error (or handle overwriting with an optional parameter).
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_EventInfo"
    exists= node in filehandle
    if(exists):
      #grab it so we can read the data
      SimEfield_EventInfo=filehandle[node]
      #create empty instance for EventIndex table
      SimEfield_EventIndex= np.zeros(1,SimEfield_EventIndex_dtype)
      #fill with the values i want
      SimEfield_EventIndex['evt_id']=SimEfield_EventInfo['evt_id']
      SimEfield_EventIndex['evt_name']=SimEfield_EventInfo['evt_name']
      SimEfield_EventIndex['n_trig']=SimEfield_EventInfo['n_trig']
      return SimEfield_EventIndex
    else:
        print("SimEfieldCompileEventIndex:Could not find ",node)
        return None


#Event Level
def SimEfieldAddEventInfo(filehandle, RunID, EventID, SimEfield_EventInfo=None):
    #
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_EventInfo"

    SimEfield_EventInfo_data= ghdf5.AddToInfo(filehandle, node, SimEfield_EventInfo_dtype,SimEfield_EventInfo )

    return SimEfield_EventInfo_data


def SimEfieldAddDetectorIndex(filehandle, RunID, EventID, SimEfield_DetectorIndex=None):
    #
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_DetectorIndex"

    SimEfield_DetectorIndex_data, item_index = ghdf5.AddToIndex(filehandle, node, SimEfield_DetectorIndex_dtype, 'det_id', SimEfield_DetectorIndex)

    return SimEfield_DetectorIndex_data, item_index

#trace level
#we might want to wrap up everything in a function that saves the trace, and updates/creates the DetectorIndex table
def SimEfieldWriteSimEfield(filehandle, RunID, EventID, DetectorID,TraceX,TraceY,TraceZ):
    #For the Trace, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_X" exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    nodeX="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_X"
    existsX = nodeX in filehandle
    nodeY="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_Y"
    existsY = nodeY in filehandle
    nodeZ="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_Z"
    existsZ = nodeZ in filehandle
    if(existsX or existsY or existsZ):
      print("SimEfieldWriteSimEfield: Event exist, not updated",RunID,EventID,DetectorID)
      return 0

    SimEfield_X_data=filehandle.create_dataset(nodeX, data=TraceX, dtype='f4')
    SimEfield_Y_data=filehandle.create_dataset(nodeY, data=TraceY, dtype='f4')
    SimEfield_Z_data=filehandle.create_dataset(nodeZ, data=TraceZ, dtype='f4')
    #return SimEfield_X_data,SimEfield_Y_data,SimEfield_Z_data

####
#Get the data from RunInfo
def SimEfieldGetRunInfo(filehandle, RunID):

    node="Run_"+str(RunID)+"/SimEfield_RunInfo"

    SimEfield_RunInfo_handle= filehandle[node]

    return SimEfield_RunInfo_handle



#Get the data from EventIndex

def SimEfieldGetEventIndex(filehandle, RunID):

    node="Run_"+str(RunID)+"/SimEfield_EventIndex"

    SimEfield_EventIndex_handle= filehandle[node]

    return SimEfield_EventIndex_handle


def SimEfieldGetNevents(SimEfield_EventIndex):

    return np.shape(SimEfield_EventIndex)[0]


def SimEfieldGetEvtID(SimEfield_EventIndex,EventNumber):
    return SimEfield_EventIndex['evt_id'][EventNumber]

def SimEfieldGetEvtName(SimEfield_EventIndex,EventNumber):

    return SimEfield_EventIndex['evt_name'][EventNumber]

#eventlevel
#Get the data from EvdentInfo
def SimEfieldGetEventInfo(filehandle, RunID, EventID):

    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_EventInfo"

    SimEfield_EventInfo_handle= filehandle[node]

    return SimEfield_EventInfo_handle


def SimEfieldGetEvtTbinsize(SimEfield_EventInfo):

    return SimEfield_EventInfo['t_bin_size']


def SimEfieldGetEvtTpre(SimEfield_EventInfo):

    return SimEfield_EventInfo['t_pre']

def SimEfieldGetEvtTpost(SimEfield_EventInfo):

    return SimEfield_EventInfo['t_post']


def SimEfieldGetTimeTrace(SimEfield_EventInfo):

 tbinsize=SimEfield.SimEfieldGetEvtTbinsize(SimEfield_EventInfo)
 tpre=SimEfield.SimEfieldGetEvtTpre(SimEfield_EventInfo)
 tpost=SimEfield.SimEfieldGetEvtTpost(SimEfield_EventInfo)



#Get the data from DetectorIndex

def SimEfieldGetDetectorIndex(filehandle, RunID, EventID):

    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimEfield_DetectorIndex"

    SimEfield_DetectorIndex_handle= filehandle[node]

    return SimEfield_DetectorIndex_handle

def SimEfieldGetNDetectors(SimEfield_DetectorIndex):

   return np.shape(SimEfield_DetectorIndex)[0]

def SimEfieldGetDetectorID(DetectorIndex,i):

   ID= DetectorIndex["det_id"][i]

   return ID.decode() #this is becouse h5py encodes in UTF-8 and returns the object as a byte value (or something like that)

def SimEfieldGetDetectorPosition(DetectorIndex,i):

   return DetectorIndex["det_pos_shc"][i]

def SimEfieldGetDetectorT0(DetectorIndex,i):

   return DetectorIndex["t_0"][i]


def SimEfieldGetEfield(filehandle,RunID,EventID,DetectorID):
    #Get it from the file
    nodeX="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_X"
    nodeY="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_Y"
    nodeZ="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/SimEfield_Z"

    tracex=filehandle[nodeX]
    tracey=filehandle[nodeY]
    tracez=filehandle[nodeZ]

    trace=np.stack((tracex,tracey,tracez),axis=1)

    return trace




