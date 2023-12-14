# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:52:11 2021

@author: QC109_3
"""
import numpy as np
import re
import yaml

# retruns how many levels of parameters there are
def get_data_level(dic, level = 1):
    if not isinstance(dic, dict) or not dic: 
        return level 
    return max(get_data_level(dic[key], level + 1) 
                               for key in dic)

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
        parameter_unit = "".join(re.findall("[^0123456789.]", output_list[0]))
        parameter_number = np.array([x.strip(parameter_unit) for x in output_list])
        return(parameter_name, parameter_number, parameter_unit)
    except (AttributeError):
        print("target level wrong")
        return()
# gets keys for data dictionary...
def make_key(name, number, unit):
    return ''.join([name, '=', number, unit])

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
        print("something's wrong with get_qubit_state()!")
    return(qubit_state)


    

f = open('Q:\Experiment_Scripts\GUI_Control_Program\SequencerProgram\SequencerData\sequencerGUI_data_210402_215355.yaml')
data = yaml.load(f)
data=data['PMT1']
f.close()
example_histogram={      0: 7,
      1: 2,
      2: 1,
      3: 0,
      4: 0,
      5: 0,
      6: 0,
      7: 0,
      8: 0,
      9: 0,
      10: 0,
      11: 0,
      12: 0,
      13: 10,
      14: 0,
      15: 0,
      16: 0,
      17: 0,
      18: 0,
      19: 0,
      20: 0,
      21: 0,
      22: 0,
      23: 0,
      24: 0,
      25: 0,
      26: 0,
      27: 0,
      28: 0,
      29: 0}

dictionary = {'key' : 'value', 'key_2': 'value_2'}
nested_dict = { 'paramA=1uu': {'paramB=3vv': 'value_1', 'paramB=4vv': 'value_2'},
                'paramA=2uu': {'paramB=3vv': 'value_1', 'paramB=4vv': 'value_2'}}

#dict_level=get_data_level(nested_dict)
#param_num, param_unit=get_parameter_values(nested_dict, dict_level, 3)

dict_level=get_data_level(data,-1)
param_name, param_num, param_unit=get_parameter_values(data, dict_level, 2)
qubit_state=get_qubit_state(example_histogram, 1)

key1=make_key(param_name, param_num[0], param_unit)
#selected_histogram=get_histogram(data, 'pos_0=10.35mm',key1)

print(dict_level)
print(param_num, param_unit)
print(qubit_state)
#print(selected_histogram)
