#Shell script for running macro CEM
#Order of inputs: interconn, co2cap

import sys,os
from MacroCEM import macroCEM

#Set working directory to location of this script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Process inputs and call master function
inputData = sys.argv[1:] #exclude 1st item (script name)
interconn = inputData[0] #ERCOT, WECC, EI
co2Cap = int(inputData[1]) #integer giving % of CO2 emissions in final year relative to first year
wsGenFracOfDemand = int(inputData[2]) #integer giving % of demand that must be supplied by wind and solar generation by final year
windGenFracOfDemand = int(inputData[3]) #integer giving % of demand that must be supplied by wind generation by final year
cesmMember = [inputData[4]] if len(inputData)>4 else None #CESM Large Ensemble member (as list)

#Other inputs
climateChange = (cesmMember != None)

macroCEM(interconn,co2Cap,wsGenFracOfDemand,windGenFracOfDemand,cesmMember,climateChange)
