import sys
import os
import glob
import logging
import numpy as np
import h5py
logging.basicConfig(level=logging.WARNING)
#
import GRANDhdf5SimEfield as SimEfield
import GRANDhdf5SimShower as SimShower
import GRANDhdf5Utilities as Util
#
#you can get ZHAIRES python from https://github.com/mjtueros/ZHAireS-Python (checkout the Development or DevelopmentLeia branch)
#I use this environment variable to let python know where to find it, but alternatively you just copy the AiresInfoFunctions.py file on the same dir you are using this.
ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
sys.path.append(ZHAIRESPYTHON)
import AiresInfoFunctions as AiresInfo


def ZHAiresRawToGRAND(HDF5handle, RunID, EventID, InputFolder,  SimEfieldInfo=True, NLongitudinal=True, ELongitudinal=True, NlowLongitudinal=True, ElowLongitudinal=True, EdepLongitudinal=True, LateralDistribution=True, EnergyDistribution=True):
    #TODO: Handle when hdf5 file does not exists
    #TODO: handle how to append an event
    #TODO: allow to specify an output filename
    #TODO: Check if the Event Exists already
    #TODO: Get the next correlative EventID if none is specified

    ShowerSimInfo=True

    idffile=glob.glob(InputFolder+"/*.idf")

    if(len(idffile)!=1):
      logging.critical("there should be one and only one idf file in the input directory!. cannot continue!")
      return -1

    sryfile=glob.glob(InputFolder+"/*.sry")

    if(len(sryfile)>1):
      logging.critical("there should be one and only one sry file in the input directory!. cannot continue!")
      return -1

    if(len(sryfile)==0):
      logging.critical("there should be one and only one sry file in the input directory!. cannot continue!")
      return -1

    EventName=AiresInfo.GetTaskNameFromSry(sryfile[0])

    inpfile=glob.glob(InputFolder+"/*.inp")
    if(len(inpfile)!=1):
      logging.critical("we can only get the core position from the input file, at it should be in the same directory as the sry")
      logging.critical("defaulting to (0.0,0)")
      inpfile=[None]
      CorePosition=(0.0,0.0,0.0)


    #############################################################################################################################
    # ShowerSimInfo (deals with the details for the simulation). This might be simulator-dependent (CoREAS has different parameters)
    #############################################################################################################################
    if(ShowerSimInfo):
        Primary= AiresInfo.GetPrimaryFromSry(sryfile[0],"GRAND")
        Zenith = AiresInfo.GetZenithAngleFromSry(sryfile[0],"GRAND")
        Azimuth = AiresInfo.GetAzimuthAngleFromSry(sryfile[0],"GRAND")
        Energy = AiresInfo.GetEnergyFromSry(sryfile[0],"GRAND")
        XmaxAltitude, XmaxDistance, XmaxX, XmaxY, XmaxZ = AiresInfo.GetKmXmaxFromSry(sryfile[0])
        XmaxAltitude= float(XmaxAltitude)*1000.0
        XmaxDistance= float(XmaxDistance)*1000.0
        XmaxPosition= [float(XmaxX)*1000.0, float(XmaxY)*1000.0, float(XmaxZ)*1000.0]
        SlantXmax=AiresInfo.GetSlantXmaxFromSry(sryfile[0])
        HadronicModel=AiresInfo.GetHadronicModelFromSry(sryfile[0])
        InjectionAltitude=AiresInfo.GetInjectionAltitudeFromSry(sryfile[0])
        Lat,Long=AiresInfo.GetLatLongFromSry(sryfile[0])
        GroundAltitude=AiresInfo.GetGroundAltitudeFromSry(sryfile[0])
        Site=AiresInfo.GetSiteFromSry(sryfile[0])
        Date=AiresInfo.GetDateFromSry(sryfile[0])
        FieldIntensity,FieldInclination,FieldDeclination=AiresInfo.GetMagneticFieldFromSry(sryfile[0])
        AtmosphericModel=AiresInfo.GetAtmosphericModelFromSry(sryfile[0])
        EnergyInNeutrinos=AiresInfo.GetEnergyFractionInNeutrinosFromSry(sryfile[0])
        EnergyInNeutrinos=EnergyInNeutrinos*Energy
        ShowerSimulator=AiresInfo.GetAiresVersionFromSry(sryfile[0])
        ShowerSimulator="Aires "+ShowerSimulator
        RandomSeed=AiresInfo.GetRandomSeedFromSry(sryfile[0])
        RelativeThinning=AiresInfo.GetThinningRelativeEnergyFromSry(sryfile[0])
        WeightFactor=AiresInfo.GetWeightFactorFromSry(sryfile[0])
        GammaEnergyCut=AiresInfo.GetGammaEnergyCutFromSry(sryfile[0])
        ElectronEnergyCut=AiresInfo.GetElectronEnergyCutFromSry(sryfile[0])
        MuonEnergyCut=AiresInfo.GetMuonEnergyCutFromSry(sryfile[0])
        MesonEnergyCut=AiresInfo.GetMesonEnergyCutFromSry(sryfile[0])
        NucleonEnergyCut=AiresInfo.GetNucleonEnergyCutFromSry(sryfile[0])
        CPUTime=AiresInfo.GetTotalCPUTimeFromSry(sryfile[0],"N/A")

        if(inpfile[0]!=None):
          CorePosition=AiresInfo.GetCorePositionFromInp(inpfile[0])

        print("CorePosition:",CorePosition)


        #create a blank RunInfo record and try to add it. If it exist, it will not change it.
        SimShower_RunInfo= np.zeros(1,SimShower.SimShower_RunInfo_dtype)
        #Populate what we can
        SimShower_RunInfo['run_id']=RunID
        SimShower_RunInfo['shower_sim']=ShowerSimulator
        SimShower_RunInfo['rel_thin']=RelativeThinning
        SimShower_RunInfo['weight_factor']=WeightFactor
        SimShower_RunInfo['lowe_cut_e']=ElectronEnergyCut
        SimShower_RunInfo['lowe_cut_gamma']=GammaEnergyCut
        SimShower_RunInfo['lowe_cut_mu']=MuonEnergyCut
        SimShower_RunInfo['lowe_cut_meson']=MesonEnergyCut
        SimShower_RunInfo['lowe_cut_nucleon']=NucleonEnergyCut
        SimShower_RunInfo_data=SimShower.SimShowerAddRunInfo(HDF5handle, RunID,SimShower_RunInfo)

        #Create the Empty EventInfo
        SimShower_EventInfo=np.zeros(1,SimShower.SimShower_EventInfo_dtype)
        #Populate what we can
        SimShower_EventInfo['run_id']=RunID
        SimShower_EventInfo['evt_id']=EventID
        SimShower_EventInfo['evt_name']=EventName
        SimShower_EventInfo['prim_energy']=Energy
        SimShower_EventInfo['prim_azimuth']=Azimuth
        SimShower_EventInfo['prim_zenith']=Zenith
        SimShower_EventInfo['prim_type']=Primary
        SimShower_EventInfo['rnd_seed']=RandomSeed
        SimShower_EventInfo['energy_in_neutrinos']=EnergyInNeutrinos
        SimShower_EventInfo['atmos_model']=AtmosphericModel
        SimShower_EventInfo['magnetic_field']=np.array([FieldInclination,FieldDeclination,FieldIntensity])
        SimShower_EventInfo['date']=Date
        SimShower_EventInfo['site']=Site
        SimShower_EventInfo['ground_alt']=GroundAltitude
        SimShower_EventInfo['site_lat_long']=np.array([Lat,Long])
        SimShower_EventInfo['prim_inj_alt']=InjectionAltitude
        SimShower_EventInfo['hadronic_model']=HadronicModel
        SimShower_EventInfo['xmax_grams']=SlantXmax
        SimShower_EventInfo['xmax_pos_shc']=np.array(XmaxPosition)
        SimShower_EventInfo['xmax_distance']=XmaxDistance
        SimShower_EventInfo['xmax_alt']=XmaxAltitude
        SimShower_EventInfo['cpu_time']=CPUTime
        SimShower_EventInfo_data=SimShower.SimShowerAddEventInfo(HDF5handle, RunID, EventID, SimShower_EventInfo)

        #grab the information to fill the EventIndex Table
        #this functionality must be provided by SimShower.
        SimShower_EventIndex=SimShower.SimShowerCompileEventIndex(HDF5handle, RunID, EventID)
        SimShower.SimShowerAddEventIndex(HDF5handle, RunID, SimShower_EventIndex)

    #############################################################################################################################
    #  SimEfieldInfo
    #############################################################################################################################
    if(SimEfieldInfo):

        #TODO: Get Refractivity Model parameters from the sry

        #Getting all the information i need for  SimEfiel
        FieldSimulator=AiresInfo.GetZHAireSVersionFromSry(sryfile[0])
        FieldSimulator="ZHAireS "+str(FieldSimulator)
        RefractionIndexModel="Exponential"
        RefractionIndexParameters=[1.0003250,-0.1218]
        print("Warning, hard coded RefractionIndexModel",RefractionIndexModel,RefractionIndexParameters)
        TimeBinSize=AiresInfo.GetTimeBinFromSry(sryfile[0])
        TimeWindowMin=AiresInfo.GetTimeWindowMinFromSry(sryfile[0])
        TimeWindowMax=AiresInfo.GetTimeWindowMaxFromSry(sryfile[0])

        #create a blank RunInfo record and try to add it. If it exist, it will not change it.
        SimEfield_RunInfo= np.zeros(1,SimEfield.SimEfield_RunInfo_dtype)
        #Populate what we can
        SimEfield_RunInfo['run_id']=RunID
        SimEfield_RunInfo['field_sim']=FieldSimulator
        SimEfield_RunInfo['refractivity_model']=RefractionIndexModel
        SimEfield_RunInfo['refractivity_param']=RefractionIndexParameters
        SimEfield_RunInfo_data=SimEfield.SimEfieldAddRunInfo(HDF5handle, RunID,SimEfield_RunInfo)

        #Create the Empty EventInfo
        SimEfield_EventInfo=np.zeros(1,SimEfield.SimEfield_EventInfo_dtype)
        #Populate what we can
        SimEfield_EventInfo['run_id']=RunID
        SimEfield_EventInfo['evt_id']=EventID
        SimEfield_EventInfo['evt_name']=EventName
        SimEfield_EventInfo['t_pre']=TimeWindowMin
        SimEfield_EventInfo['t_post']=TimeWindowMax
        SimEfield_EventInfo['t_bin_size']=TimeBinSize
        SimEfield_EventInfo_data=SimEfield.SimEfieldAddEventInfo(HDF5handle, RunID, EventID, SimEfield_EventInfo)

        #Go through the available antennas and CreateAndFill SimEfieldDetectorIndex
        IDs,antx,anty,antz,antt=AiresInfo.GetAntennaInfoFromSry(sryfile[0])



        antx=np.array(antx, dtype=np.float32)
        anty=np.array(anty, dtype=np.float32)
        antz=np.array(antz, dtype=np.float32)
        antt=np.array(antt, dtype=np.float32)

        #ZHAIRES DEPENDENT
        ending_e = "/a*.trace"
        tracefiles=glob.glob(InputFolder+ending_e)

        if(len(tracefiles)==0):
         logging.critical("no trace files found in "+showerdirectory+" ZHAireSHDF5FileWriter cannot continue")

        for ant in tracefiles:

            ant_number = int(ant.split('/')[-1].split('.trace')[0].split('a')[-1]) # index in selected antenna list. this only works if all antenna files are consecutive

            DetectorID = IDs[ant_number]
            ant_position=(antx[ant_number],anty[ant_number],antz[ant_number])

            efield = np.loadtxt(ant,dtype='f4') #we read the electric field as a numpy array

            #create and Empty imEfield_DetectorIndex
            SimEfield_DetectorIndex=np.zeros(1,SimEfield.SimEfield_DetectorIndex_dtype)
            #Populate what we can
            SimEfield_DetectorIndex['det_id' ]=DetectorID
            SimEfield_DetectorIndex['det_pos_shc']=ant_position
            SimEfield_DetectorIndex['det_type']="ZHAireS"
            SimEfield_DetectorIndex['t_0']=antt[ant_number]
            SimEfield_DetectorIndex_data,antennaindex=SimEfield.SimEfieldAddDetectorIndex(HDF5handle,RunID,EventID, SimEfield_DetectorIndex)

            TraceX=efield[:,1]
            TraceY=efield[:,2]
            TraceZ=efield[:,3]
            SimEfield.SimEfieldWriteSimEfield(HDF5handle, RunID, EventID, DetectorID,TraceX,TraceY,TraceZ)


        #grab the information to fill the EventIndex Table
        #this functionality must be provided by SimEfield.
        SimEfield_EventIndex=SimEfield.SimEfieldCompileEventIndex(HDF5handle, RunID, EventID)
        SimEfield.SimEfieldAddEventIndex(HDF5handle, RunID, SimEfield_EventIndex)


    ##############################################################################################################################
    # LONGITUDINAL TABLES
    ##############################################################################################################################

    if(NLongitudinal):
        #the gammas table
        table=AiresInfo.GetLongitudinalTable(InputFolder,1001,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteSlantDepth(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteNgammas(HDF5handle, RunID, EventID, table.T[1])

        #the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,1205,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteVerticalDepth(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteNeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the e plus (yes, the positrons)
        table=AiresInfo.GetLongitudinalTable(InputFolder,1006,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNeplus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus mu minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1207,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1007,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNmuplus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus pi munus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1211,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNpiplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1011,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNpiplus(HDF5handle, RunID, EventID, table.T[1])

        #and the all charged
        table=AiresInfo.GetLongitudinalTable(InputFolder,1291,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNallcharged(HDF5handle, RunID, EventID, table.T[1])

    ##############################################################################################################################
    # Energy LONGITUDINAL TABLES (very important to veryfy the energy balance of the cascade, and to compute the invisible energy)
    ##############################################################################################################################
    if(ELongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,1501,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,1705,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the mu plus mu minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1707,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the pi plus pi minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1711,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEpiplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the k plus k minus
        table=AiresInfo.GetLongitudinalTable(InputFolder,1713,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEkplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the neutrons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1521,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEneutrons(HDF5handle, RunID, EventID, table.T[1])

        #the protons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1522,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEprotons(HDF5handle, RunID, EventID, table.T[1])

        #the anti-protons
        table=AiresInfo.GetLongitudinalTable(InputFolder,1523,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEpbar(HDF5handle, RunID, EventID, table.T[1])

        #the nuclei
        table=AiresInfo.GetLongitudinalTable(InputFolder,1541,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEnuclei(HDF5handle, RunID, EventID, table.T[1])

        #the other charged
        table=AiresInfo.GetLongitudinalTable(InputFolder,1591,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEother_charged(HDF5handle, RunID, EventID, table.T[1])

        #the other neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,1592,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEother_neutral(HDF5handle, RunID, EventID, table.T[1])

        #and the all
        table=AiresInfo.GetLongitudinalTable(InputFolder,1793,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEall(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # NLowEnergy Longitudinal development
    #################################################################################################################################
    if(NlowLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7001,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteNlowgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7005,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteNloweplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7006,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteNloweplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7207,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteNlowmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7091,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteNlowother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7092,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteNlowother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # ELowEnergy Longitudinal development
    #################################################################################################################################
    if(ElowLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7501,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteElowgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7505,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEloweplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7506,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEloweplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7707,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteElowmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7591,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteElowother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7592,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteElowother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # EnergyDeposit Longitudinal development
    #################################################################################################################################
    if(EdepLongitudinal):
        #the gammas
        table=AiresInfo.GetLongitudinalTable(InputFolder,7801,Slant=True,Precision="Simple")
        SimShower.SimShowerWriteEdepgammas(HDF5handle, RunID, EventID, table.T[1])

        #i call the eplusminus table, in vertical, to store also the vertical depth
        table=AiresInfo.GetLongitudinalTable(InputFolder,7805,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEdepeplusminus(HDF5handle, RunID, EventID, table.T[1])

        #the positrons (note that they will deposit twice their rest mass!)
        table=AiresInfo.GetLongitudinalTable(InputFolder,7806,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEdepeplus(HDF5handle, RunID, EventID, table.T[1])

        #the muons
        table=AiresInfo.GetLongitudinalTable(InputFolder,7907,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEdepmuons(HDF5handle, RunID, EventID, table.T[1])

        #Other Chaged
        table=AiresInfo.GetLongitudinalTable(InputFolder,7891,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEdepother_charged(HDF5handle, RunID, EventID, table.T[1])

        #Other Neutral
        table=AiresInfo.GetLongitudinalTable(InputFolder,7892,Slant=False,Precision="Simple")
        SimShower.SimShowerWriteEdepother_neutral(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # Lateral Tables
    #################################################################################################################################
    if(LateralDistribution):
        #the gammas
        table=AiresInfo.GetLateralTable(InputFolder,2001,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFradius(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteLDFgamma(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2205,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFeplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2006,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFeplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2207,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFmuplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2007,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFmuplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2291,Density=False,Precision="Simple")
        SimShower.SimShowerWriteLDFallcharged(HDF5handle, RunID, EventID, table.T[1])

    ################################################################################################################################
    # Energy Distribution at ground Tables
    #################################################################################################################################
    if(EnergyDistribution):
        #the gammas
        table=AiresInfo.GetLateralTable(InputFolder,2501,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_energy(HDF5handle, RunID, EventID, table.T[0])
        SimShower.SimShowerWriteEnergyDist_gammas(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2705,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_eplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2506,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_eplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2707,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_muplusminus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2507,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_muplus(HDF5handle, RunID, EventID, table.T[1])

        table=AiresInfo.GetLateralTable(InputFolder,2791,Density=False,Precision="Simple")
        SimShower.SimShowerWriteEnergyDist_allcharged(HDF5handle, RunID, EventID, table.T[1])

    return EventName

if __name__ == '__main__':

  if (len(sys.argv)>6 or len(sys.argv)<6) :
    print("Please point me to a directory with some ZHAires output, and indicate the mode RunID, EventID and output filename...nothing more, nothing less!")
    print("i.e ZHAiresRawToGRAND ./MyshowerDir full RunID EventID MyFile.hdf5")
    mode="exit"

  elif len(sys.argv)==6 :
    inputfolder=sys.argv[1]
    mode=sys.argv[2]
    RunID=int(sys.argv[3])
    EventID=int(sys.argv[4])
    FileName=sys.argv[5]
    HDF5handle= h5py.File(FileName, 'a')

  if(mode=="standard"):
      ZHAiresRawToGRAND(HDF5handle,RunID,EventID,inputfolder)

  elif(mode=="full"):

      ZHAiresRawToGRAND(HDF5handle,RunID,EventID,inputfolder, SimEfieldInfo=True, NLongitudinal=True, ELongitudinal=True, NlowLongitudinal=True, ElowLongitudinal=True, EdepLongitudinal=True, LateralDistribution=True, EnergyDistribution=True)

  elif(mode=="minimal"):

      ZHAiresRawToGRAND(HDF5handle,RunID,EventID,inputfolder,  SimEfieldInfo=True, NLongitudinal=False, ELongitudinal=False, NlowLongitudinal=False, ElowLongitudinal=False, EdepLongitudinal=False, LateralDistribution=False, EnergyDistribution=False)


  else:

      print("please enter one of these modes: standard, full or minimal")

