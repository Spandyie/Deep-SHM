# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:33:18 2018

@author: Spandan
"""

from scipy.io import loadmat
import numpy as np


class Ace:
    _numPath=1
    
    def __init__(self):
        self._data=None
        self.sensor={}#dict.fromkeys(range(41))
        self.actuator={}#dict.fromkeys(range(41))
        self.setup={}
        self.setup['sampling_rate'] = None
        self.setup['sensor_range'] = None
        self.setup['actuator_range'] = None
        self.setup['number_data_points'] = None
        self.setup['number_pretrigger'] = None
        #self.temperature = dict.fromkeys(range(10))
        self.setup['signal_definition']={}
        self.setup['signal_definition']['actuator'] = []
        self.setup['signal_definition']['sensor'] = []
        self.setup['signal_definition']['amplitude'] = []
        self.setup['signal_definition']['frequency1'] = []
        self.setup['signal_definition']['frequency2'] = []
        self.setup['signal_definition']['signal_type'] = []
        self.setup['signal_definition']['act_nbr_points'] = []
        self.setup['signal_definition']['act_sampling_frequency'] = []
        self.setup['signal_definition']['user_signal_filename'] = []
        self.setup['signal_definition']['gain'] = []
#        self.setup['temperature']= np.ones((2,10)) * -1000
        
     
              
    def load(self,filename):
#        DfnFile,shmDiagInfo = Ace._loadDfn()                     
        self._data = loadmat(filename,squeeze_me = False, verify_compressed_data_integrity=True)
        numSensors =1
#        numSensors = int(DfnFile[0].split(" ")[1])
#        assert(numSensors== (len(DfnFile)-1))
        
        for i in range(numSensors):
            self.sensor["s" +str(i)] = self._data["s"+str(i)]
            self.actuator["a" + str(i)] = self._data["s"+str(i)]
        self.setup['sampling_rate'] = int(self._data['setup']['sampling_rate'])
        self.setup['sensor_range'] = int(self._data['setup']['sensor_range'])
        self.setup['actuator_range'] = int(self._data['setup']['actuator_range'])
        self.setup['number_data_points'] = int(self._data['setup']['number_data_points'])
        self.setup['number_pretrigger'] = int(self._data['setup']['number_pretrigger'])        
#        m , n = self._data['setup']['temperature'][0][0].shape
#        for i in range(m):
#            for j in range(n):
#                self.setup['temperature'][i][j] = self._data['setup']['temperature'][0][0][i][j]        
        number_of_paths = len(self._data['setup']['signal_definition'][0][0][0])
        
        for x in range(number_of_paths):
            self.setup['signal_definition']['actuator'].append(int(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][0])))
            self.setup['signal_definition']['sensor'].append(int(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][1])))
            self.setup['signal_definition']['amplitude'].append(int(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][2])))
            self.setup['signal_definition']['signal_type'].append(self._data['setup']['signal_definition'][0][0][0][x][3][0][0][0])
            self.setup['signal_definition']['frequency1'].append(int(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][4])))
            self.setup['signal_definition']['frequency2'].append(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][5]))            
            self.setup['signal_definition']['act_nbr_points'].append(self._data['setup']['signal_definition'][0][0][0][x][6])
            self.setup['signal_definition']['act_sampling_frequency'].append(self._data['setup']['signal_definition'][0][0][0][x][7])
            self.setup['signal_definition']['user_signal_filename'].append(self._data['setup']['signal_definition'][0][0][0][x][8])
            self.setup['signal_definition']['gain'].append(int(np.squeeze(self._data['setup']['signal_definition'][0][0][0][x][9])))
            
    
#        
#    def _loadAce(self):
#        with open("Cure_Vac_SL2\\Cure_Vac_SL2.ace","r") as acefile:
#            AceFile=[]
#            for line in acefile:
#                AceFile.append(line[:-1])
#        return AceFile
    
#    @staticmethod
#    def _loadDfn():
#        with open("SHM_diag_info.txt") as shm_diag:
#            shmDiagInfo = shm_diag.readlines()           
#                  
#        with open(shmDiagInfo[-1][:-1],"r") as dfile:
#            DfnFile=[]
#            for line in dfile:
#                DfnFile.append(line[:-1])
#        return DfnFile, shmDiagInfo