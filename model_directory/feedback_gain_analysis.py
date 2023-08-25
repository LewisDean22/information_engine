# -*- coding: utf-8 -*-
"""

Zero trap work condition implemented by gain = 2

@author: Lewis Dean
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

import model_constants as constants
from power_calculator import calculate_mean_powers

DATA_DIRECTORY = "data_files/"
SAVE_DATA = 1
NUMBER_OF_POINTS = 40

MIN_GAIN = 0
MAX_GAIN = 4
TAU_C = 10**-4  # nonequilibrium white noise limit
D_NE = 0 # Zero nonequilibrium noise
DELTA_G = 0.37
    

def main(tau_c=TAU_C, d_ne=D_NE, delta_g=DELTA_G,
         model=constants.NOISE_MODEL):
    
    gain_linspace = np.linspace(MIN_GAIN, MAX_GAIN, num=NUMBER_OF_POINTS)
    
    p = Pool()
    args = [(tau_c, d_ne, gain, delta_g) for gain in gain_linspace]
    result_array  = p.starmap(calculate_mean_powers, args)
    result_array = np.array(result_array)
    mean_free_powers = result_array[:,0]
    std_mean_free_powers = result_array[:,1]
    mean_trap_powers = result_array[:,2]
    std_mean_trap_powers = result_array[:,3]
    
    mean_net_power = mean_free_powers - mean_trap_powers
    std_mean_net_power = np.sqrt(std_mean_free_powers**2 + 
                                 std_mean_trap_powers**2)
        
    if SAVE_DATA:
        
        file = np.zeros((len(gain_linspace), 7))
        file[:,0] = gain_linspace
        file[:,1] = mean_free_powers
        file[:,2] = std_mean_free_powers
        file[:,3] = mean_trap_powers
        file[:,4] = std_mean_trap_powers
        file[:,5] = mean_net_power
        file[:,6] = std_mean_net_power
        headers = ["feedback_gain", "mean_free_power",
                   "std_mean_free_power", "mean_trap_power",
                   "std_mean_trap_power", "mean_net_power",
                   "std_mean_net_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY + 
                   "power_against_feedback_gain_model_{}.csv".format(
                       model)), index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
