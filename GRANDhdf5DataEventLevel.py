import h5py
import os
import sys
import numpy as np
import GRANDhdf5Utilities as ghdf5

# Types
# AntennaInfo
AntennaInfo_dtype = np.dtype ([
                                ('evt_antenna_id','i4'),
                                ('evt_gps_sec','i8'),
                                ('evt_nanosec','i8'),
                                ('evt_trigger_flag','i4'),
                                ('evt_year','i4'),
                                ('evt_month','i4'),
                                ('evt_day','i4'),
                                ('evt_hour','i4'),
                                ('evt_minute','i4'),
                                ('evt_second','i4'),
                                ('evt_elec_status','i4'),
                                ('evt_ctd','i8'),
                                ('evt_gps_quant','f4',(1,2)),
                                ('evt_ctp','i8'),
                                ('evt_synchronization','i4'),
                                ('evt_temperature','f4')
])

# EventHeader
EventHeader_dtype = np.dtype ([
                                ('evt_run_nr','i4'),
                                ('evt_event_nr','i4'),
                                ('evt_t3__nr','i4'),
                                ('evt_second','i8'),
                                ('evt_nanosec','i8'),
                                ('evt_n_detector','i4')
])

# Methods
def DataEventAddAntennaInfo(filehandle, RunID, EventID, AntennaInfo=None):

    node="Run_" + str(RunID) + "/Event_" + str(EventID) + "/AntennaInfo"

    AntennaInfo_data, item_index = ghdf5.AddToIndex(filehandle, node, AntennaInfo_dtype, 'evt_antenna_id',AntennaInfo)

    return AntennaInfo_data, item_index


def DataEventAddEventHeader(filehandle, RunID, EventID, EventHeader=None ):

    node="Run_" + str(RunID) + "/Event_" + str(EventID) + "/EventHeader"

    EventHeader_data= ghdf5.AddToInfo(filehandle, node, EventHeader_dtype, EventHeader )

    return EventHeader_data

def DataEventAddTraces(filehandle, RunID, EventID, AntennaID,TraceX,TraceY,TraceZ):
    #For the Trace, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+DetectorID+"/ADC_X" exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    nodeX="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+AntennaID+"/ADC_X"
    existsX = nodeX in filehandle
    nodeY="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+AntennaID+"/ADC_Y"
    existsY = nodeY in filehandle
    nodeZ="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/Traces_"+AntennaID+"/ADC_Z"
    existsZ = nodeZ in filehandle
    if(existsX or existsY or existsZ):
      print("DataAddTraces: Event exist, not updated",RunId,EventID,DetectorID)
      return 0

    ADC_X_data=filehandle.create_dataset(nodeX, data=TraceX, dtype='f4')
    ADC_Y_data=filehandle.create_dataset(nodeY, data=TraceY, dtype='f4')
    ADC_Z_data=filehandle.create_dataset(nodeZ, data=TraceZ, dtype='f4')
    #return ADC_X_data,ADC_Y_data,ADC_Z_data
