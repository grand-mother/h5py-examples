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
#print(type(SimShower_DetectorInfo_data))
#print(type(SimShower_DetectorInfo_data['det_id'])) #this gives me access to the column (a numpy array)
#print(type(SimShower_DetectorInfo_data['det_id'][:])) #this gives me the contents of the column (a numpy array)
#print(type(SimShower_DetectorInfo_data['det_id'][0])) #and this gives me the first element (of the content, not access to the file)
#print(type(SimShower_DetectorInfo_data[0,'det_id'])) #so, this gives me access to the 'det_id' field of record 0
#print(SimShower_DetectorInfo_data[0])

#SimShower RunLevelInfo
SimShower_RunInfo_dtype =np.dtype  ([('run_id', 'u8'),           #RunID: Just to be sure we are in the right place
                                     ('shower_sim','S20'),       #Name of the simulator (with version)
                                     ('rel_thin','f4'),          #Relative thinning energy
                                     ('weight_factor','f4'),
                                     ('resampling_ratio','f4'),
                                     ('lowe_cut_e','f4'),
                                     ('lowe_cut_mu','f4'),
                                     ('lowe_cut_gamma','f4'),
                                     ('lowe_cut_meson','f4'),
                                     ('lowe_cut_nucleon','f4')
                                     ])

#SimShower RunLevelIndex
SimShower_EventIndex_dtype =np.dtype  ([('evt_id', 'u8'),    #EventID: Just to be sure we are in the right place
                                      ('evt_name','S100'), #ZHAireS TaskName usefull to keep to find the original files
                                      ('hadronic_model','S20'),   #Name of the hadronic model (with version)
                                      ('prim_energy','f4'),
                                      ('prim_type','f4'),
                                      ('prim_zenith','f4'),
                                      ('prim_azimuth','f4'),
                                      ('xmax_distance','f8'), #
                                      ('xmax_grams','f4'),              #Xmax in grams. (single precision)
                                      ('prim_core','f8',(1, 4))
                                     ])

#SimShower EventLevelInfo
SimShower_EventInfo_dtype=np.dtype([('evt_id', 'u8'),    #EventID: This has to be defined. An unsigned int?
                                    ('run_id', 'u8'),
                                    ('evt_name','S100'), #ZHAireS TaskName usefull to keep to find the original files
                                    ('rnd_seed', 'f8'),
                                    ('hadronic_model','S20'),   #Name of the hadronic model (with version)
                                    ('prim_energy','f4'),             #Energy in 32bit (single precision)(GeV)
                                    ('prim_type','S20'),
                                    #site
                                    ('date','S12'),
                                    ('site','S20'),
                                    ('site_lat_long','f4',(1,2)),
                                    ('ground_alt','f4'),
                                    ('magnetic_field','f4',(1,3)),     #Inclination, Declination (deg) and Field strenght in uT
                                    ('atmos_model','S20'),
                                    ('atmos_model_param','f4',(1,3)),
                                    #Geometry parameters
                                    ('prim_azimuth','f4'),            #
                                    ('prim_zenith','f4'),
                                    ('prim_injpoint_shc','f8',(1, 4)),#Include time in the 4th column? or make a new variable? 64bits, cos we want cm precision at 10E6m scale.
                                    ('prim_core','f8',(1, 4)),        #Include time in the 4th column? or make a new variable? 64bits, cos we want cm precision at 10E6m scale.
                                    ('prim_inj_dir','f8',(1,3)),      #redundant, but handy.
                                    ('prim_inj_alt','f4'),            #redundant, but handy.
                                    #FirstLevelShowerReconstructedParameters (What you can get just after the sim)
                                    ('xmax_pos_shc','f8',(1,3)),      #In meters. Needs to be accurate to 10 centimeter in 1000km
                                    ('xmax_grams','f4'),              #Xmax in grams. (single precision)
                                    ('xmax_distance','f8'),           #redundant, but handy.
                                    ('xmax_alt','f8'),                #redundant, but handy.
                                    ('gh_fit_param','f4',(1,3)),      #Store the rest of the GH fit parameters lambda,X0, and Chi2.
                                    ('energy_in_neutrinos','f4'),     #Energy in neutrinos produced on the shower (GeV). Needed for computing the invisible energy
                                    ('cpu_time','f4',(1,3))           #in seconds
                                   ])
#Longitudinal Profile Tables

#Lateral Profile Tables

#Run Level
def SimShowerAddRunInfo(filehandle, RunID,SimShower_RunInfo=None ):

    node="Run_"+str(RunID)+"/SimShower_RunInfo"

    SimShower_RunInfo_data= ghdf5.AddToInfo(filehandle, node, SimShower_RunInfo_dtype,SimShower_RunInfo )

    return SimShower_RunInfo_data


def SimShowerAddEventIndex(filehandle, RunID,SimShower_EventIndex=None):

    node="Run_"+str(RunID)+"/SimShower_EventIndex"

    SimShower_EventIndex_data, item_index = ghdf5.AddToIndex(filehandle, node, SimShower_EventIndex_dtype, 'evt_id', SimShower_EventIndex)

    return SimShower_EventIndex_data, item_index

#this is the function that compiles the information of the event that will go to the EventIndex
def SimShowerCompileEventIndex(filehandle, RunID, EventID):

    #Information that we will extract from the SimShower_EventInfo
    #check if "Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimShower_EventInfo" already exist. If it does, give an error (or handle overwriting with an optional parameter).
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimShower_EventInfo"
    exists= node in filehandle
    if(exists):
      #grab it so we can read the data
      SimShower_EventInfo=filehandle[node]
      #create empty instance for EventIndex table
      SimShower_EventIndex= np.zeros(1,SimShower_EventIndex_dtype)
      #fill with the values i want
      SimShower_EventIndex['evt_id']=SimShower_EventInfo['evt_id']
      SimShower_EventIndex['evt_name']=SimShower_EventInfo['evt_name']
      SimShower_EventIndex['hadronic_model']=SimShower_EventInfo['hadronic_model']
      SimShower_EventIndex['prim_energy']=SimShower_EventInfo['prim_energy']
      SimShower_EventIndex['prim_type']=SimShower_EventInfo['prim_type']
      SimShower_EventIndex['prim_zenith']=SimShower_EventInfo['prim_zenith']
      SimShower_EventIndex['prim_azimuth']=SimShower_EventInfo['prim_azimuth']
      SimShower_EventIndex['xmax_distance']=SimShower_EventInfo['xmax_distance']
      SimShower_EventIndex['xmax_grams']=SimShower_EventInfo['xmax_grams']
      SimShower_EventIndex['prim_core']=SimShower_EventInfo['prim_core']
      return SimShower_EventIndex
    else:
        print("SimShowerCompileEventIndex:Could not find ",node)
        return None


#Event Level
def SimShowerAddEventInfo(filehandle, RunID, EventID, SimShower_EventInfo=None):
    #
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/SimShower_EventInfo"

    SimShower_EventInfo_data= ghdf5.AddToInfo(filehandle, node, SimShower_EventInfo_dtype,SimShower_EventInfo )

    return SimShower_EventInfo_data



#tables
def SimShowerWriteLongTable(filehandle, RunID, EventID, TableName, TableData):
    #For the Tables, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if table exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/LongTables"+"/"+str(TableName)
    exists = node in filehandle

    if(exists):
      print("SimShowerWriteLongTable: Event exist, not updated",RunID,EventID,TableName)
      return 0

    SimShower_table=filehandle.create_dataset(node, data=TableData, dtype='f4')

#number of particles
def SimShowerWriteSlantDepth(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "SlantDepth", TableData)

def SimShowerWriteVerticalDepth(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "VerticalDepth", TableData)

def SimShowerWriteNgammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_gammas", TableData)

def SimShowerWriteNeplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_eplusminus", TableData)

def SimShowerWriteNeplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_eplus", TableData)

def SimShowerWriteNmuplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_muplusminus", TableData)

def SimShowerWriteNmuplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_muplus", TableData)

def SimShowerWriteNpiplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_piplusminus", TableData)

def SimShowerWriteNpiplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_piplus", TableData)

def SimShowerWriteNallcharged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "N_allcharged", TableData)

#energy tables
def SimShowerWriteEgammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_gammas", TableData)

def SimShowerWriteEeplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_eplusminus", TableData)

def SimShowerWriteEmuplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_muplusminus", TableData)

def SimShowerWriteEpiplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_piplusminus", TableData)

def SimShowerWriteEkplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_kplusminus", TableData)

def SimShowerWriteEneutrons(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_neutrons", TableData)

def SimShowerWriteEprotons(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_protons", TableData)

def SimShowerWriteEpbar(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_pbar", TableData)

def SimShowerWriteEnuclei(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_nuclei", TableData)

def SimShowerWriteEother_charged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_other_charged", TableData)

def SimShowerWriteEother_neutral(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_other_neutral", TableData)

def SimShowerWriteEall(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "E_all", TableData)

#N low energy tables
def SimShowerWriteNlowgammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_gammas", TableData)

def SimShowerWriteNloweplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_eplusminus", TableData)

def SimShowerWriteNloweplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_eplus", TableData)

def SimShowerWriteNlowmuons(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_muons", TableData)

def SimShowerWriteNlowother_charged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_other_charged", TableData)

def SimShowerWriteNlowother_neutral(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Nlow_other_neutral", TableData)

#E low energy tables
def SimShowerWriteElowgammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_gammas", TableData)

def SimShowerWriteEloweplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_eplusminus", TableData)

def SimShowerWriteEloweplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_eplus", TableData)

def SimShowerWriteElowmuons(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_muons", TableData)

def SimShowerWriteElowother_charged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_other_charged", TableData)

def SimShowerWriteElowother_neutral(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Elow_other_neutral", TableData)

#E dep tables
def SimShowerWriteEdepgammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_gammas", TableData)

def SimShowerWriteEdepeplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_eplusminus", TableData)

def SimShowerWriteEdepeplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_eplus", TableData)

def SimShowerWriteEdepmuons(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_muons", TableData)

def SimShowerWriteEdepother_charged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_other_charged", TableData)

def SimShowerWriteEdepother_neutral(filehandle, RunID, EventID, TableData):
  SimShowerWriteLongTable(filehandle, RunID, EventID, "Edep_other_neutral", TableData)

def SimShowerWriteLateralTable(filehandle, RunID, EventID, TableName, TableData):
    #For the Tables, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if table exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/LateralTables"+"/"+str(TableName)
    exists = node in filehandle

    if(exists):
      print("SimShowerWriteLateralTable: Event exist, not updated",RunID,EventID,TableName)
      return 0

    SimShower_table=filehandle.create_dataset(node, data=TableData, dtype='f4')

def SimShowerWriteLDFradius(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_radius", TableData)

def SimShowerWriteLDFgamma(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_gammas", TableData)

def SimShowerWriteLDFeplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_eplusminus", TableData)

def SimShowerWriteLDFeplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_eplus", TableData)

def SimShowerWriteLDFmuplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_muplusminus", TableData)

def SimShowerWriteLDFmuplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_muplus", TableData)

def SimShowerWriteLDFallcharged(filehandle, RunID, EventID, TableData):
  SimShowerWriteLateralTable(filehandle, RunID, EventID, "LDF_allcharged", TableData)


def SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, TableName, TableData):
    #For the Tables, it makes no sense to go through the logic of creating an empty record and then filling it up, becouse we will either have the trace and wanto store it, or we wont.
    #
    #check file exists, is open for writing/appending. If it does not exist give an error
    #check if table exists, if it does, give an error (or handle overwriting with an optional parameter)
    #Put it on the file
    node="Run_"+str(RunID)+"/"+"Event_"+str(EventID)+"/EnergyDistTables"+"/"+str(TableName)
    exists = node in filehandle

    if(exists):
      print("SimShowerWriteEnergyDistTable: Event exist, not updated",RunID,EventID,TableName)
      return 0

    SimShower_table=filehandle.create_dataset(node, data=TableData, dtype='f4')


def SimShowerWriteEnergyDist_energy(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "Energy", TableData)

def SimShowerWriteEnergyDist_gammas(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_gammas", TableData)

def SimShowerWriteEnergyDist_eplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_eplusminus", TableData)

def SimShowerWriteEnergyDist_eplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_eplus", TableData)

def SimShowerWriteEnergyDist_muplusminus(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_muplusminus", TableData)

def SimShowerWriteEnergyDist_muplus(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_muplus", TableData)

def SimShowerWriteEnergyDist_allcharged(filehandle, RunID, EventID, TableData):
  SimShowerWriteEnergyDistTable(filehandle, RunID, EventID, "EnergyDist_allcharged", TableData)




def SimShowerGetEventIndex(filehandle, RunID):

    node="Run_"+str(RunID)+"/SimShower_EventIndex"

    SimShower_EventIndex_handle= filehandle[node]

    return SimShower_EventIndex_handle


def SimShowerGetNevents(SimShower_EventIndex):

    return np.shape(SimShower_EventIndex)[0]


def SimShowerGetEvtID(SimShower_EventIndex,EventNumber):
    return SimShower_EventIndex['evt_id'][EventNumber]

def SimShowerGetEvtName(SimShower_EventIndex,EventNumber):

    return SimShower_EventIndex['evt_name'][EventNumber]

def SimShowerGetEvtZenith(SimShower_EventIndex,EventNumber):

    return SimShower_EventIndex['prim_zenith'][EventNumber]

def SimShowerGetEvtAzimuth(SimShower_EventIndex,EventNumber):

    return SimShower_EventIndex['prim_azimuth'][EventNumber]







