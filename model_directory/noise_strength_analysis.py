# -*- coding: utf-8 -*-
"""

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

D_NE_MIN_POWER = -2
D_NE_MAX_POWER = 2
TAU_C = 10**-4
GAIN = 2
DELTA_G = 0.37
    

def main(delta_g=DELTA_G, tau_c=TAU_C, gain=GAIN,
         model=constants.NOISE_MODEL):

    
    d_ne_logspace = np.logspace(D_NE_MIN_POWER, D_NE_MAX_POWER,
                                num=NUMBER_OF_POINTS, base=10)
                                  
    p = Pool()
    arguments = [(tau_c, d_ne, gain, delta_g) for d_ne in d_ne_logspace] 
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
        
        file = np.zeros((len(d_ne_logspace), 7))
        file[:,0] = d_ne_logspace
        file[:,1] = mean_free_powers
        file[:,2] = std_mean_free_powers
        file[:,3] = mean_trap_powers
        file[:,4] = std_mean_trap_powers
        file[:,5] = mean_net_power
        file[:,6] = std_mean_net_power
        headers = ["noise_strength", "mean_free_power",
                   "std_mean_free_power", "mean_trap_power",
                   "std_mean_trap_power", "mean_net_power",
                   "std_mean_net_power"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY +
                  "power_against_noise_strength_gain_{0}_model_{1}.csv".format(
                      gain, model)), index=0, header=1, sep=',')
    
  
if __name__ == '__main__':
    main()
