# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Created on Tue Jul 19 21:52:20 2023

NEED TO ADD UNITS TO VERTICAL AXIS

@author: lewis
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

import model_constants as constants
from power_calculator import calculate_mean_powers


IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"

SAVE_PLOT = 1
SAVE_DATA = 1

SAVE_DATA = 1
NUMBER_OF_POINTS = 40

DELTA_G_MIN = 0
DELTA_G_MAX = 3
TAU_C = 10**-4
D_NE = 3
GAIN = 2
    

def main(tau_c=TAU_C, d_ne=D_NE, gain=GAIN, model=constants.NOISE_MODEL):

    
    delta_g_linspace = np.linspace(DELTA_G_MIN, DELTA_G_MAX,
                                   num=NUMBER_OF_POINTS)

    p = Pool()
    arguments = [(tau_c, d_ne, gain, delta_g) for delta_g in delta_g_linspace] 
    result_array = p.starmap(calculate_mean_powers, arguments)   
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_mean_free_powers = result_array[:,1]
    mean_trap_powers = result_array[:,2]
    std_mean_trap_powers = result_array[:,3]
    
    mean_net_power = mean_free_powers - mean_trap_powers
    std_mean_net_power = np.sqrt(std_mean_free_powers**2 + 
                                 std_mean_trap_powers**2)
        
    if SAVE_DATA:
        
        file = np.zeros((len(delta_g_linspace), 7))
        file[:,0] = delta_g_linspace
        file[:,1] = mean_free_powers
        file[:,2] = std_mean_free_powers
        file[:,3] = mean_trap_powers
        file[:,4] = std_mean_trap_powers
        file[:,5] = mean_net_power
        file[:,6] = std_mean_net_power
        headers = ["delta_g", "mean_free_power",
                   "std_mean_free_power", "mean_trap_power",
                   "std_mean_trap_power", "mean_net_power",
                   "std_mean_net_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY +
                  "power_against_delta_g_gain_{0}_model_{1}.csv".format(
                      gain, model)), index=0, header=1, sep=',')

    
if __name__ == '__main__':
    main()
