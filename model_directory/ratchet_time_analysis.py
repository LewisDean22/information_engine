# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:05:10 2023

@author: lewis
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

from ratchet_time_generator import mean_first_passage_time
import model_constants as constants

DATA_DIRECTORY = "data_files/"
SAVE_DATA = 1
NUMBER_OF_POINTS = 10

D_NE = 3
GAIN = 2
DELTA_G = 0.37

TAU_C_MIN_POWER = -3
TAU_C_MAX_POWER = 3
D_NE = 3
GAIN = 2
DELTA_G = 0.37


def main(d_ne=D_NE, gain=GAIN, delta_g=DELTA_G,
         model=constants.NOISE_MODEL):
    
    tau_c_logspace = np.logspace(TAU_C_MIN_POWER, TAU_C_MAX_POWER,
                                 num=NUMBER_OF_POINTS, base=10)
    
    p = Pool()
    args = [(tau_c, d_ne, gain, delta_g) for tau_c in tau_c_logspace]
    
    result_array  = p.starmap(mean_first_passage_time, args)
    result_array = np.array(result_array)
    mean_ratchet_times = result_array[:,0]
    std_mean_ratchet_times = result_array[:,1]
    
    if SAVE_DATA:
        
        file = np.zeros((len(tau_c_logspace), 3))
        file[:,0] = tau_c_logspace
        file[:,1] = mean_ratchet_times
        file[:,2] = std_mean_ratchet_times
        headers = ["correlation_time", "mean_ratchet_time",
                   "std_mean_ratchet_time"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY + 
                   "ratchet_time_against_correlation_time_gain_{0}_model_{1}.csv".format(
                       gain, model)), index=0, header=1, sep=',')
        
    return None


if __name__ == "__main__":
    main()
