import os
import sys
import logging
import numpy as np
import h5py
#root_dir = os.path.realpath(os.path.join(os.path.split(__file__)[0], "../radio-simus")) # = $PROJECT
root_dir=os.environ["RADIOSIMUS"] #this requires the radi simus package from grand
sys.path.append(os.path.join(root_dir, "lib", "python"))
from radio_simus.in_out import _table_voltage
from radio_simus.computevoltage import compute_antennaresponse
from radio_simus.signal_processing import filters
import GRANDhdf5Utilities as ghdf5
import GRANDhdf5SimEfield as SimEfield
import GRANDhdf5SimSignal as SimSignal
import GRANDhdf5SimShower as SimShower

logging.basicConfig(level=logging.DEBUG)

#if no outfilename is given, it will store the table in the same HDF5 dile, in a separate table (TODO: handle what happens if it already exists)

#this computes the voltage on all the antennas

def ComputeVoltageOnHDF5(inputfilename,RunID=0,outfilename="N/A"):
#EventNumber=all could trigger a loop on all events in the file.

  if os.path.isfile(inputfilename):
    if(outfilename=="N/A"):
      infilehandle=h5py.File(inputfilename, 'a')
      outfilehandle=infilehandle
    else:
      infilehandle=h5py.File(inputfilename, 'r')
      outfilehandle=h5py.File(outfilename, 'a')

    RunInfo_hanlde=SimEfield.SimEfieldGetRunInfo(infilehandle,RunID)
    ShowerEventIndex_handle=SimShower.SimShowerGetEventIndex(infilehandle,RunID)
    ShowerEventIndex=ShowerEventIndex_handle[:] #this actually loads the ShowerEventIndex (all of it)

    NumberOfEvents=SimShower.SimShowerGetNevents(ShowerEventIndex)

    logging.info("Computing Voltages for "+inputfilename+", found "+str(NumberOfEvents)+" events")


    for idx in range(0,NumberOfEvents):

        EventID=SimShower.SimShowerGetEvtID(ShowerEventIndex,idx)
        EventName=SimShower.SimShowerGetEvtName(ShowerEventIndex,idx)
        Zenith=SimShower.SimShowerGetEvtZenith(ShowerEventIndex,idx)
        Azimuth=SimShower.SimShowerGetEvtAzimuth(ShowerEventIndex,idx)

        print(type(Zenith),type(Azimuth),Zenith,Azimuth)


        DetectorIndex_handle=SimEfield.SimEfieldGetDetectorIndex(infilehandle,RunID,EventID)
        DetectorIndex=DetectorIndex_handle[:] #this actually loads the DetectorIndex (all of it)

        nantennas=SimEfield.SimEfieldGetNDetectors(DetectorIndex)
        logging.info("Found "+str(nantennas)+" antennas")

        EventInfo_handle=SimEfield.SimEfieldGetEventInfo(infilehandle, RunID, EventID)
        EventInfo=EventInfo_handle[:]

        tbinsize=SimEfield.SimEfieldGetEvtTbinsize(EventInfo)
        tpre=SimEfield.SimEfieldGetEvtTpre(EventInfo)
        tpost=SimEfield.SimEfieldGetEvtTpost(EventInfo)


        for i in range(0,nantennas):

          DetectorID=SimEfield.SimEfieldGetDetectorID(DetectorIndex,i)

          logging.info("computing voltage for antenna "+DetectorID+" ("+str(i+1)+"/"+str(nantennas)+")")

          position=SimEfield.SimEfieldGetDetectorPosition(DetectorIndex,i)
          logging.debug("at position"+str(position))

          efield=SimEfield.SimEfieldGetEfield(infilehandle,RunID,EventID,DetectorID)

          t0=SimEfield.SimEfieldGetDetectorT0(DetectorIndex,i)

          time=np.arange(tpre+t0,tpost+t0+10*tbinsize,tbinsize,)

          time=time[0:np.shape(efield)[0]]

          efield=np.column_stack((time,efield))

          #i compute the antenna response using the compute_antennaresponse function
          #A NICE call to the radio-simus library. Configuration and details of the voltage computation unavailable for now!.
          #Configuration should be a little more "present" in the function call,
          #also maybe the library to handle .ini files would be more profesional and robust than current implementation

          voltage = compute_antennaresponse(efield, Zenith, Azimuth, alpha=0, beta=0 )

          #now i need to put a numpy array into an astropy table, but before y change the data type to float32 so that it takes less space (its still good to 7 decimals)
          voltage32= voltage.astype('f4')

          SimSignal.SimSignalWriteSimSignal(outfilehandle, RunID, EventID, DetectorID,voltage32[:,1],voltage32[:,2],voltage32[:,3])

        #end for

  else:
   logging.critical("input file " + inputfilename + " does not exist or is not a directory. ComputeVoltageOnSHDF5 cannot continue")


if __name__ == '__main__':

  if ( len(sys.argv)<2 ):
    print("usage ComputVoltagaOnHDF5 inputfile (outputfile)")
    print("if outputfile is not specified, voltage is writen on the same file")

  if ( len(sys.argv)==2 ):
   inputfile=sys.argv[1]
   ComputeVoltageOnHDF5(inputfile)

  if ( len(sys.argv)==3 ):
   inputfile=sys.argv[1]
   outputfile=sys.argv[2]
   ComputeVoltageOnHDF5(inputfile,outfilename=outputfile)




