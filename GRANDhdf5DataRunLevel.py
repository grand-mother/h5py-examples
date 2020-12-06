import h5py
import os
import sys
import numpy as np
import GRANDhdf5Utilities as ghdf5

# Types
#Data CenterField
CenterField_dtype = np.dtype  ([
                                ('center_latitude','f4'),
                                ('center_longitude','f4'),
                                ('center_altitude','f4'),
                                ('center_x','f4'),
                                ('center_y','f4')
])

#Data DetectorInfo
DetectorInfo_dtype = np.dtype ([
                                ('run_antenna_id','i4'),
                                ('run_latitude','f4'),
                                ('run_longitude','f4'),
                                ('run_altitude','f4'),
                                ('run_x','f4'),
                                ('run_y','f4'),
                                ('run_antenna_model','S100'),
                                ('run_electronics_id','i4'),
                                ('run_electronics_model','S100')
])

#Data ElectronicsSettings
ElectronicsSettings_dtype = np.dtype ([
                                        ('run_electronics_id','i4'),
                                        ('run_trigger_mask','i4'),
                                        ('run_trace_lengths','i4',(1,4)),
                                        ('run_thresholds','i4',(4,2)),
                                        ('run_serial_version','i4'),
                                        ('run_longitude','f4'),
                                        ('run_latitude','f4'),
                                        ('run_altitude','f4'),
                                        ('run_control','i4'),
                                        ('run_trigger_enable','i4'),
                                        ('run_channel_mask','i4'),
                                        ('run_trigger_divider','i4'),
                                        ('run_coincidence_readout','i4'),
                                        ('run_ctrl','i4'),
                                        ('run_pre_post_length','i4',(4,2)),
])

# Methods
def DataRunAddCenterField(filehandle, RunID, CenterField=None ):

    node="Run_"+str(RunID)+"/CenterField"

    CenterField_data= ghdf5.AddToInfo(filehandle, node, CenterField_dtype, CenterField )

    return CenterField_data

def DataRunAddDetectorInfo(filehandle, RunID, DetectorInfo=None):

    node="Run_"+str(RunID)+"/DetectorInfo"

    DetectorInfo_data, item_index = ghdf5.AddToIndex(filehandle, node, DetectorInfo_dtype, 'run_antenna_id',DetectorInfo)

    return DetectorInfo_data, item_index

def DataRunAddElectronicsSettings(filehandle, RunID, ElectronicsSettings=None):

    node="Run_"+str(RunID)+"/ElectronicsSettings"

    ElectronicsSettings_data, item_index = ghdf5.AddToIndex(filehandle, node, ElectronicsSettings_dtype, 'run_electronics_id',ElectronicsSettings)

    return ElectronicsSettings_data, item_index

