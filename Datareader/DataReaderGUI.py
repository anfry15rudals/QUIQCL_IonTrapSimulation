# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:55:09 2020

@author: Honggi Jeon
@email: sl725@snu.ac.kr
"""

from __future__ import unicode_literals
from pathlib import Path
import os, sys
filename = os.path.abspath(__file__)
dirname = os.path.dirname(filename)
root_dirname=Path(dirname).parents[1]   #chamber folder

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.optimize import curve_fit
import yaml
import re
import copy


qt_designer_file = dirname + '\\DataReader_GUI_v1.ui'
uifile, QtBaseClass = uic.loadUiType(qt_designer_file)

####################CONSTANTS AND DEFINITIONS GO HERE###########################
# retruns how many levels of parameters there are. arg level is offset. when level=-1, level 0 is the photon count histogram
def get_data_level(dic, level = -1):
    if not isinstance(dic, dict) or not dic: 
        return level 
    return max(get_data_level(dic[key], level + 1) 
                               for key in list(dic))

# given a sequencer data file which is a nested dictionary, goes to target level and return parameter numbers as a numpy array and unit as a string
def get_parameter_values(dic, initial_level, target_level):
    current_level=initial_level
    input_dict=dic
    if initial_level==target_level:
        output_dict=input_dict
    else:
        while current_level>target_level:
            first_key=list(input_dict.keys())[0]
            output_dict=input_dict[first_key]
            input_dict=output_dict
            current_level-=1
    try:
        output_list=list(output_dict.keys())
        marker_loc=output_list[0].index('=')
        parameter_name = output_list[0][:marker_loc]
        output_list=[x[marker_loc+1:] for x in output_list]
        parameter_unit = "".join(re.findall("[^-0123456789.]", output_list[0]))
        if not parameter_unit:   #if parameter unit is emtpy, which is when sequencer is running in continuous mode or scan parameter is integer
            parameter_number = np.array([int(x.strip(parameter_unit)) for x in output_list])
        else:
            parameter_number = np.array([float(x.strip(parameter_unit)) for x in output_list])
        return(parameter_name, parameter_number, parameter_unit)
    except (AttributeError):
        print("target level wrong")
        return()
# gets keys for data dictionary...
def make_key(name, number, unit):
    return ''.join([name, '=', str(number), unit])

# returns histogram given up to two keys. needs to be fixed if more than 2 parameters need to be scanned  
def get_histogram(dic, key_a, key_b=None):
    if key_b is None:   #if data is 1D scan
        histogram_dict=dic[key_a]
    else:
        histogram_dict=dic[key_a][key_b]
    return(histogram_dict)
    
# given a photon count histogram and discrimination condition, returns the probability of |1> state by counting dark events(probably faster...)
def get_qubit_state(histogram_dict, threshold):
    photon_numbers=np.array(list(histogram_dict.keys()))
    dark_events=0
    total_events=0
    for histogram_key in photon_numbers:
        total_events+=histogram_dict[histogram_key]
        if histogram_key<threshold:
            dark_events+=histogram_dict[histogram_key]
    try:
        qubit_state=1-dark_events/total_events
    except:
#        print("something's wrong with get_qubit_state()! returning -1")
        return(-1)
    return(qubit_state)

#get photon counts by summing over all entries of histogram and subtracting background, which is 0 if not given
def get_photon_counts(histogram_dict, get_average=False, bg_counts=0):
    photon_numbers=np.array(list(histogram_dict.keys()))
    event_numbers=np.array(list(histogram_dict.values()))
    total_event_number=event_numbers.sum()
    if not get_average:
        photon_counts=np.inner(photon_numbers, event_numbers)-bg_counts
    elif get_average:
        photon_counts=(np.inner(photon_numbers, event_numbers)-bg_counts)/total_event_number
    return photon_counts

#prepare state axes for 1D scan
def prepare_axes_1D(axes_obj, y_bound, y_label, x_bound, x_label, title_str):
    axes_obj.set_ylim(y_bound)
    axes_obj.set_ylabel(y_label)
    if min(x_bound)==max(x_bound):
        x_bound = [0, 1]
    axes_obj.set_xlim(x_bound)
    axes_obj.set_xlabel(x_label)
    axes_obj.set_title(title_str)
    return() 
################################################################################

class DataReaderGUI(QMainWindow, uifile):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.initUi()
        self.initActions()
        self.setGeometry(100, 100, 1600, 900)

    def initUi(self):
        #declare default values
        self.DATAFILE_PATH = str(root_dirname) + "\\GUI_Control_Program\SequencerData"
        self.param_a_unit, self.param_b_unit="parameter a", "parameter b"
        self.DATAFILE_NAME="no file"
        self.data=None
        self.GRAPH_MARGIN=1.2
        self.THRESHOLD=3.5
        self.paramA_name, self.paramA_number, self.paramA_unit, self.paramA_selected=np.array([]), np.array([]), "unit", 0
        self.paramB_name, self.paramB_number, self.paramB_unit, self.paramB_selected=np.array([]), np.array([]), "unit", 0
        self.qubit_state, self.photon_counts, self.average_photon_counts= np.array([]), np.array([]), np.array([])
        self.data_level=-1
        self.key_a, key_b="", ""
        self.FIT_AMPL_INIT, self.FIT_PHASE_INIT=1.0, 0.0 
        self.FIT_FREQ_INIT, self.FIT_OFFSET_INIT=0.01, 0   #fit frequency is in MHz, since parameter scan is in [us]
        self.FIT_RESULTS=[0, 0, 0, 0]
        #initialize gui values
        self.text_threshold.setText(str(self.THRESHOLD))
        self.text_fitampl.setText(str(self.FIT_AMPL_INIT))
        self.text_fitphase.setText(str(self.FIT_PHASE_INIT))
        self.text_fitfreq.setText(str(self.FIT_FREQ_INIT))
        self.text_fitoffset.setText(str(self.FIT_OFFSET_INIT))
        self.radio_qubitstate.setChecked(True)
        #create widgets
        self.fig_hist = plt.Figure()
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.toolbar_hist = NavigationToolbar(self.canvas_hist, self)
        self.ax_hist = self.fig_hist.add_subplot(1,1,1)
        self.fig_state = plt.Figure()
        self.canvas_state = FigureCanvas(self.fig_state)
        self.ax_state = self.fig_state.add_subplot(1,1,1)
        self.toolbar_state = NavigationToolbar(self.canvas_state, self)
        self.Plot_Box.addWidget(self.toolbar_hist)
        self.Plot_Box.addWidget(self.canvas_hist)
        self.Plot_Box.addWidget(self.toolbar_state)
        self.Plot_Box.addWidget(self.canvas_state)
        #initialize axes and labels
        self.update_labels()
        self.update_state_axes()
        self.update_hist_axes()        

    def initActions(self):  #Define GUI element actions
        self.button_load.clicked.connect(self.load_data)
        self.button_load.clicked.connect(self.update_labels)
        self.button_load.clicked.connect(self.update_state_axes)
        self.button_load.clicked.connect(self.update_hist_axes)
        
        self.button_sinefit.clicked.connect(self.sinusoidal_fit)
        self.button_sinefit.clicked.connect(self.update_labels)
        
        self.radio_qubitstate.toggled.connect(self.update_state_axes) 
        self.radio_counts.toggled.connect(self.update_state_axes)
        self.radio_avg_counts.toggled.connect(self.update_state_axes)
        
        self.button_apply.clicked.connect(self.change_threshold)
        self.button_apply.clicked.connect(self.update_state_axes)
        self.button_apply.clicked.connect(self.update_hist_axes)

        self.text_fitampl.textChanged.connect(self.fit_init_changed)
        self.text_fitphase.textChanged.connect(self.fit_init_changed)
        self.text_fitfreq.textChanged.connect(self.fit_init_changed)

        self.canvas_state.mpl_connect('pick_event', self.select_data_point)

    def load_data(self):
        extension_filter = "yaml(*.yaml)"
        title="Select Data"
        datafile_name = QFileDialog.getOpenFileName(self, title, self.DATAFILE_PATH, extension_filter)
        try:
            f = open(datafile_name[0])
            data = yaml.load(f, Loader=yaml.SafeLoader)
            f.close()
            self.DATAFILE_NAME=datafile_name[0]
            self.data=data
            self.data_handler(self.data)
        except:
            pass
        
    def data_handler(self, data=None, PMTnum_handled=0):
       if not data:
           return()
       else:
           if not PMTnum_handled:
               data=data['PMT1']   #this needs to change to support more than 1 input
           key_a, key_b=None, None
           qubit_state=np.array([])
           photon_counts=np.array([])
           average_photon_counts=np.array([])
#           copy_data = copy.deepcopy(data)
           data_level=get_data_level(data)
           
           if data_level==1:#if one parameter is scanned
                paramA_name, paramA_number, paramA_unit=get_parameter_values(data, data_level, data_level)  # get the outer most scan parameter and save as paramA...
                for a in paramA_number:
                    key_a=make_key(paramA_name, a, paramA_unit)
                    histogram_dict=data[key_a]
                    qubit_state=np.append(qubit_state, get_qubit_state(histogram_dict, self.THRESHOLD))
                    photon_counts=np.append(photon_counts, get_photon_counts(histogram_dict))
                    average_photon_counts=np.append(average_photon_counts, get_photon_counts(histogram_dict, get_average=True))
                self.paramA_name, self.paramA_number, self.paramA_unit=paramA_name, paramA_number, paramA_unit
                self.qubit_state, self.photon_counts, self.average_photon_counts=qubit_state, photon_counts, average_photon_counts
           elif data_level==2:
                paramA_name, paramA_number, paramA_unit=get_parameter_values(data, data_level, data_level)  # get the outermost scan parameter and save as paramA...
                paramB_name, paramB_number, paramB_unit=get_parameter_values(data, data_level, data_level-1)# get the second outermost scan parameter and save as paramA...
                for a in paramA_number:
                    for b in paramB_number:
                        key_a=make_key(paramA_name, a, paramA_unit)
                        key_b=make_key(paramB_name, b, paramB_unit)
                        histogram_dict=data[key_a][key_b]
                        qubit_state=np.append(qubit_state, get_qubit_state(histogram_dict, self.THRESHOLD))
                        photon_counts=np.append(photon_counts, get_photon_counts(histogram_dict))
                        average_photon_counts=np.append(average_photon_counts, get_photon_counts(histogram_dict, get_average=True))
                qubit_state=np.reshape(qubit_state, (len(paramA_number), len(paramB_number)))
                photon_counts=np.reshape(photon_counts, (len(paramA_number), len(paramB_number)))
                average_photon_counts=np.reshape(average_photon_counts, (len(paramA_number), len(paramB_number)))
                self.paramA_name, self.paramA_number, self.paramA_unit=paramA_name, paramA_number, paramA_unit
                self.paramB_name, self.paramB_number, self.paramB_unit=paramB_name, paramB_number, paramB_unit
                self.qubit_state, self.photon_counts, self.average_photon_counts=qubit_state, photon_counts, average_photon_counts
           else:
                print("more than 2 scan parameters not supported yet")
           self.data_level=data_level
           self.data=data

    def update_state_axes(self, index_a=0, index_b=0):
        index_a=int(index_a)
        index_b=int(index_b)

        self.ax_state.cla()
        if self.qubit_state.size==0:    #if qubit_state is empty
            prepare_axes_1D(self.ax_state, [0, 1], 'P1', [0, 100], '['+self.paramA_unit+']','No Data Yet!')
        elif self.data_level==1:    #if one parameter is scanned
            key_a=make_key(self.paramA_name, self.paramA_number[index_a], self.paramA_unit)
            x_label='['+self.paramA_unit+']'
            if self.radio_qubitstate.isChecked():
                X, Z= self.paramA_number, self.qubit_state
                title_str=key_a+",   "+str(self.qubit_state[index_a])
                prepare_axes_1D(self.ax_state, [0, 1], 'P1', [min(X), max(X)], x_label, title_str)
            elif self.radio_counts.isChecked():
                X, Z= self.paramA_number, self.photon_counts
                title_str=key_a+",   "+str(self.photon_counts[index_a])
                prepare_axes_1D(self.ax_state, [0, max(Z)*self.GRAPH_MARGIN], 'Total Photon Counts', [min(X), max(X)], x_label, title_str)
            elif self.radio_avg_counts.isChecked():
                X, Z= self.paramA_number, self.average_photon_counts
                title_str=key_a+",   "+str(self.average_photon_counts[index_a])
                prepare_axes_1D(self.ax_state, [0, max(Z)*self.GRAPH_MARGIN], 'Average Photon Counts per Shot', [min(X), max(X)], x_label, title_str)
            self.ax_state.plot(X, Z, 'ro', picker=4)
            self.ax_state.grid(True)
        elif self.data_level==2:
            key_a=make_key(self.paramA_name, self.paramA_number[index_a], self.paramA_unit)
            key_b=make_key(self.paramB_name, self.paramB_number[index_b], self.paramB_unit)
            self.ax_state.set_ylabel('['+self.paramA_unit+']')      
            self.ax_state.set_xlabel('['+self.paramB_unit+']')
            if self.radio_qubitstate.isChecked():
                self.ax_state.set_title("Qubit State"+", "+key_a+", "+key_b+",   "+str(self.qubit_state[index_a][index_b]))
                Z=np.flipud(self.qubit_state)
            elif self.radio_counts.isChecked():
                self.ax_state.set_title("Photon Counts"+", "+key_a+", "+key_b+",   "+str(self.photon_counts[index_a][index_b]))
                Z=np.flipud(self.photon_counts)
            elif self.radio_avg_counts.isChecked():
                self.ax_state.set_title("Average Photon Counts"+", "+key_a+", "+key_b+",   "+str(self.average_photon_counts[index_a][index_b]))
                Z=np.flipud(self.average_photon_counts)
            if len(self.paramA_number)<2:
                stepA = 0
            else:
                stepA = (self.paramA_number[1]-self.paramA_number[0])
            if len(self.paramB_number)<2:
                stepB = 0
            else:
                stepB = (self.paramB_number[1]-self.paramB_number[0])
            self.ax_state.imshow(Z, extent=(np.amin(self.paramB_number)-stepB/2, np.amax(self.paramB_number)+stepB/2,
                                            np.amin(self.paramA_number)-stepA/2, np.amax(self.paramA_number)+stepA/2), picker=4)
        self.ax_state.set_aspect('auto')
        self.fig_state.tight_layout()
        self.canvas_state.draw()
        
    def update_hist_axes(self, index_a = 0, index_b = 0):
        index_a=int(index_a)
        index_b=int(index_b)

        self.ax_hist.cla()
        self.ax_hist.set_xlim(0, 100)
        self.ax_hist.set_ylim([0, 10])
        if self.data_level==1:    #if one parameter is scanned
            key_a=make_key(self.paramA_name, self.paramA_number[index_a], self.paramA_unit)
            histogram_dict=self.data[key_a]
            self.ax_hist.set_title(key_a+", "+str(get_photon_counts(histogram_dict)))
        elif self.data_level==2:    #if two parameters are scanned
            key_a=make_key(self.paramA_name, self.paramA_number[index_a], self.paramA_unit)
            key_b=make_key(self.paramB_name, self.paramB_number[index_b], self.paramB_unit)
            histogram_dict=self.data[key_a][key_b]
            self.ax_hist.set_title(key_a+", "+key_b+", "+str(get_photon_counts(histogram_dict)))
        else: 
            histogram_dict=dict(zip(range(30),[0]*30))  #use list of zeros for initialization when no data is loaded
        X=np.array(list(histogram_dict.keys()))
        Z=np.array(list(histogram_dict.values()))
        self.ax_hist.hist(X, bins=np.array(range(max(X)+1))-0.5, weights=Z)
        self.ax_hist.set_xticks(range(max(X)+1))
        self.ax_hist.set_xlim([-0.5, max(X)*self.GRAPH_MARGIN-0.5])
        self.ax_hist.set_ylim([0, max(max(Z), 0.01)*self.GRAPH_MARGIN])
        self.ax_hist.axvline(x=self.THRESHOLD, linewidth=4, linestyle='--', color='r')
        self.ax_hist.set_xlabel('# of photons detected')         
        self.ax_hist.set_ylabel('# of events')
        self.ax_state.set_aspect('auto')
        self.fig_state.tight_layout()
        self.canvas_hist.draw()

    def select_data_point(self, event):
        cursor_x = event.mouseevent.xdata
        cursor_y = event.mouseevent.ydata
        if self.data_level==1:    #if one parameter is scanned
            if self.radio_qubitstate.isChecked():           
                distances = np.hypot(cursor_x - self.paramA_number, cursor_y - self.qubit_state)
            elif self.radio_counts:
                distances = np.hypot(cursor_x -  self.paramA_number, cursor_y - self.photon_counts)
            elif self.radio_avg_counts:
                distances = np.hypot(cursor_x -  self.paramA_number, cursor_y - self.average_photon_counts)
            selected_index_a = distances.argmin()
            self.update_hist_axes(selected_index_a)
            self.update_state_axes(selected_index_a)
        elif self.data_level==2:    #if two parameters are scanned
            meshB, meshA=np.meshgrid(self.paramB_number, self.paramA_number)
            distances = np.hypot(cursor_x - meshA, cursor_y - meshB)
            selected_index_a, selected_index_b = np.unravel_index(distances.argmin(), distances.shape)
            self.update_hist_axes(selected_index_b, selected_index_a)
            self.update_state_axes(selected_index_b, selected_index_a)
        
    def update_data_reader(self, data):
        self.data_handler(data)
        self.update_state_axes()
        self.update_hist_axes()
    
    def update_labels(self):
        self.label_filename.setText(self.DATAFILE_NAME)
        self.label_fitresults.setText("ampl: %.2f, freq: %.4f MHZ\n phase: %.2f, offset: %.2f"
                                      % (self.FIT_RESULTS[0], self.FIT_RESULTS[1], self.FIT_RESULTS[2], self.FIT_RESULTS[3]))

    def change_threshold(self):
        threshold_str=self.text_threshold.text()
        try:
            threshold=float(threshold_str)
            self.THRESHOLD=threshold
            self.data_handler(self.data, PMTnum_handled=1)
        except:
            print("wrong threshold")
            return()

    def fit_init_changed(self):
        self.FIT_AMPL_INIT=float(self.text_fitampl.text())
        self.FIT_PHASE_INIT=float(self.text_fitphase.text())
        self.FIT_FREQ_INIT=float(self.text_fitfreq.text())
        self.FIT_FREQ_OFFSET=float(self.text_fitoffset.text())
        
    def sinusoidal_fit(self):
        def sine_model(x, a, f, p, c):
            return a*np.sin(np.pi*f*x+p)**2+c
        LOWER_BOUNDS, UPPER_BOUNDS=[0, 0, -np.pi, -0.5], [1.5, 10, np.pi, 0.5]  #freq bounds in MHz
        fit_results, _ = curve_fit(sine_model, self.param_a, self.P1_data,
                        bounds=(LOWER_BOUNDS, UPPER_BOUNDS), 
                        p0=[self.FIT_AMPL_INIT, self.FIT_PHASE_INIT, self.FIT_FREQ_INIT, self.FIT_OFFSET_INIT])
        
        X=np.linspace(self.param_a[0], self.param_a[-1], 1000)
        Y=sine_model(X, fit_results[0], fit_results[1], fit_results[2], fit_results[3])
        self.ax_state.plot(X, Y, color='r', ls='--', alpha=0.5)
        self.canvas_state.draw()
        self.FIT_RESULTS=fit_results
        


####################OBSOLATE FUNCTIONS###########################
"""
    def get_P1(self):
        P1_dict=dict()
        
        if self.data_depth==0:  #if single parameter is scanned
            for i in self.hist_dict.keys():
                one_freq, total_freq=0, 0
                for count in self.hist_dict[i]:
                    if count>self.THRESHOLD:
                        one_freq+=self.hist_dict[i][count]
                    total_freq+=self.hist_dict[i][count]   
                total_freq=max(0.1, total_freq)
                P1_dict[i]=one_freq/total_freq
            if P1_dict:
                _, self.P1_data=zip(*sorted(zip(list(P1_dict.keys()), list(P1_dict.values()))))
        else:   #more than 1 parameter is scanned
            return()

    def update_state(self):
        self.ax_state.cla()
        A=[float(k) for k in self.param_a]
        B=[float(k) for k in self.param_b]
        photon_counts=list(self.counts_dict.values())
        
        if self.P1_data.size==0:    #if P1_data is empty
            self.ax_state.set_xlim([0, 100])
            self.ax_state.set_xlabel('['+self.param_a_unit+']')
            self.ax_state.set_ylim([0, 1])
            self.ax_state.set_ylabel('P1')
        elif self.data_depth==0:    #if single parameter is scanned
            if self.radio_qubitstate.isChecked():
                self.ax_state.set_ylim([0, 1])
                self.ax_state.set_ylabel('P1')
                self.ax_state.set_title(str(self.key_a)+",   "+str(self.P1_picked))
                Y=self.P1_data
            else:
                self.ax_state.set_ylim([0, max(photon_counts)*self.GRAPH_MARGIN])
                self.ax_state.set_ylabel('Total Photon Counts')
                self.ax_state.set_title(str(self.key_a)+",   "+str(self.Counts_picked))
                Y=photon_counts
            self.ax_state.plot(A, Y, 'ro-', picker=1)
            self.ax_state.set_xlim([min(A), max(A)])
            self.ax_state.set_xlabel('['+self.param_a_unit+']')
        else:   #if more than one parameter is scanned
            if self.radio_qubitstate.isChecked():
                self.ax_state.set_ylabel('P1')
                self.ax_state.set_title(str(self.key_a) + ",   " + ",   " + str(self.key_b) + str(self.P1_picked))
                Y=self.P1_data
            else:
                self.ax_state.set_ylim([0, max(self.counts_dict.values())*self.GRAPH_MARGIN])
                self.ax_state.set_ylabel('Total Photon Counts')
                self.ax_state.set_title(str(self.key_a) + ",   " + ",   " + str(self.key_b) + str(self.Counts_picked))
                Y=self.counts_dict.values()
            self.ax_state.imshow(Y, extent=(np.amin(A), np.amax(A), np.amin(B), np.amax(B)))
            self.ax_state.set_xlabel('['+self.param_a_unit+']')
        self.canvas_state.draw()
        
    def update_hist(self):
        self.ax_hist.cla()
        if not self.hist_dict:  #if histogram dictionary is empty
            self.ax_hist.set_xlim(0, 100)
            self.ax_hist.set_ylim([0, 10])
        elif self.data_depth>0:   #more than 1d scan
            self.ax_hist.set_title(str(self.key_a) + ",   " + ",   " + str(self.key_b)+'['+self.param_a_unit+']')
        else:    #1d scan
            X=list(self.hist_dict[str(self.hist_key)].keys())
            Y=list(self.hist_dict[str(self.hist_key)].values())
            self.ax_hist.set_title(str(self.key_a)+'['+self.param_a_unit+']')
            self.ax_hist.hist(X, bins=range(max(X)+1), weights=Y)
            self.ax_hist.set_xticks(range(max(X)+1))
            self.ax_hist.set_xlim([0, max(X)*self.GRAPH_MARGIN])
            self.ax_hist.set_ylim([0, max(Y)*self.GRAPH_MARGIN])
        self.ax_hist.axvline(x=self.THRESHOLD, linewidth=4, linestyle='--', color='r')
        self.ax_hist.set_xlabel('# of photons detected')         
        self.ax_hist.set_ylabel('# of events')
        self.canvas_hist.draw()

    def state_point_picked(self, event):

        if self.radio_qubitstate.isChecked():           
            distances = np.hypot(x - param_a_float, y - self.P1_data)
            indmin = distances.argmin()
            self.hist_key, self.P1_picked=self.param_a[indmin], self.P1_data[indmin]
        else:
            distances = np.hypot(x - param_a_float, y - list(self.counts_dict.values()))
            indmin = distances.argmin()
            self.hist_key, self.Counts_picked=self.param_a[indmin], list(self.counts_dict.values())[indmin]
        self.update_state()
        self.update_hist()
"""

if __name__ == '__main__':
    app=0
    app = QApplication(sys.argv)
    w = DataReaderGUI()
    w.show()
    app.exec_()
    sys.exit(app.exec())