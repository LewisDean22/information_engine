# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:54:36 2023

@author: lewis
"""
from data_plotter import tau_c_gaussian_bath_powers

# Chosen parameters for inputted tau_c data
TAU_C_DELTA_G = 0.37


def main(delta_g=TAU_C_DELTA_G):
    
    gaussian_bath_power = tau_c_gaussian_bath_powers(2, delta_g)
    print("Gaussian bath (low limit) power = {}".format(gaussian_bath_power))
    
    return None


if __name__ == "__main__":
    main()