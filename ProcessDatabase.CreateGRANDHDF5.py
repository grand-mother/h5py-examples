import sys
import os
import logging   #for...you guessed it...logging
import sqlite3   #for the database
import argparse  #for command line parsing
import h5py      #for hdf5 file opening

#hack to import ZHAireS-Runner using environment variables
ZHAIRESRUNNER=os.environ["ZHAIRESRUNNER"]
sys.path.append(ZHAIRESRUNNER + "/TheLibraryRunner/Scripts") #so that it knows where to find things
print(ZHAIRESRUNNER + "/TheLibraryRunner/Scripts")
import DatabaseFunctions as mydatabase  #my database handling library

#hack to import ZHAireS-Python using environment variables
ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
sys.path.append(ZHAIRESPYTHON)
import AiresInfoFunctions as AiresInfo

#hack to import the desired python inrerpreter using environment variables
PYTHONINTERPRETER=os.environ["PYTHONINTERPRETER"]

import ZHAireSRawToGRANDHDF5 as ZRTG


#Remember that in the database paths are relative, so you need to run this in the directory where the database is

parser = argparse.ArgumentParser(description='A script to get the CPU time in a library of Simulations')
parser.add_argument('DatabaseFile', #name of the parameter
                    metavar="dbfile", #name of the parameter value in the help
                    help='The Database of the library .db file') # help message for this parameter
parser.add_argument('StartRecord', #name of the parameter
                    metavar="start", #name of the parameter value in the help
                    help='Record on wich to start (inclusive)') # help message for this parameter
parser.add_argument('EndRecord', #name of the parameter
                    metavar="end", #name of the parameter value in the help
                    help='Record on wich to stop (inclusive)') # help message for this parameter
parser.add_argument('RunID', #name of the parameter
                    metavar="runid", #name of the parameter value in the help
                    help='RunID under wich all the events of this call will be grouped ') # help message for this parameter


results = parser.parse_args()
dbfile=results.DatabaseFile
start=int(results.StartRecord)
end=int(results.EndRecord)
runid=results.RunID

#logging.debug('This is a debug message')
#logging.info('This is an info message')
#logging.warning('This is a warning message')
#logging.error('This is an error message')
#logging.critical('This is a critical message')
logging.basicConfig(level=logging.DEBUG)


logging.debug("Starting Creation of GRAND HDF5 Files on database %s " % dbfile)
DataBase=mydatabase.ConnectToDataBase(dbfile)
#this is to show the current status of the database
mydatabase.GetDatabaseStatus(DataBase)
#This is how you search on the database, here im selecting everything (To Do: functions to search the database)
#This is to get a cursor on the database. You can think of the cursor as a working environment. You can have many cursors.
CurDataBase = DataBase.cursor()
CurDataBase.execute("SELECT * FROM showers")

OutputFile=runid+".GRAND.hdf5"
HDF5handle= h5py.File(OutputFile, 'a')

DatabaseRecord=CurDataBase.fetchone()
while(DatabaseRecord!=None):

  DatabaseStatus=mydatabase.GetStatusFromRecord(DatabaseRecord) #i do it with a function call becouse if we change the database structure we dont have to change this
  Directory=mydatabase.GetDirectoryFromRecord(DatabaseRecord)
  JobName=mydatabase.GetNameFromRecord(DatabaseRecord)
  Id=mydatabase.GetIdFromRecord(DatabaseRecord)
  TaskName=mydatabase.GetTasknameFromRecord(DatabaseRecord)

  if(Id>=start and Id <=end and DatabaseStatus=="RunOK"):
    logging.debug(str(Id) + " Reading Job " + JobName + " which was in " + DatabaseStatus + " status at " + Directory)

    sryfile= str(Directory)+"/"+str(TaskName)+".sry"

    print(sryfile,Id)

    if(os.path.isfile(sryfile)):
       idftarfile= str(Directory)+"/"+str(TaskName)+".idf.tar.gz"
       cmd="tar -xzvf "+ idftarfile
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       tracestarfile= str(Directory)+"/"+str(TaskName)+".traces.tar.gz"
       cmd="tar -xzvf "+ tracestarfile +" >/dev/null 2>&1"
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       sryfile=str(Directory)+"/"+str(TaskName)+".sry"
       cmd="cp "+sryfile+" ."
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       #Runnign the shit
       cmd=PYTHONINTERPRETER + "ZHAireSRawToGRANDHDF5.py"
       #out = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
       #stdout,stderr = out.communicate()
       RunID=runid
       EventID=TaskName
       InputFolder="."
       #ZRTG.ZHAiresRawToGRAND(HDF5handle, RunID, EventID, InputFolder,  SimEfieldInfo=True, NLongitudinal=True, ELongitudinal=True, NlowLongitudinal=True, ElowLongitudinal=True, EdepLongitudinal=True, LateralDistribution=True, EnergyDistribution=True)
       ZRTG.ZHAiresRawToGRAND(HDF5handle, RunID, EventID, InputFolder,  SimEfieldInfo=True, NLongitudinal=False, ELongitudinal=False, NlowLongitudinal=False, ElowLongitudinal=False, EdepLongitudinal=False, LateralDistribution=False, EnergyDistribution=False)
       #cleaning
       cmd="rm "+str(TaskName)+".sry"
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       cmd="rm "+str(TaskName)+".idf"
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       cmd="rm a*.trace"
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       cmd="rm antpos.dat"
       logging.debug("about to run:" + cmd )
       os.system(cmd)

       cmd="rm "+str(TaskName)+".lgf"
       logging.debug("about to run:" + cmd )
       os.system(cmd)



  #this is the last order of the while, that will fetch the next record of the database
  DatabaseRecord=CurDataBase.fetchone()


