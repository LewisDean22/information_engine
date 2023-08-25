# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Noise model 1: Coloured Ornstein-Uhlenbeck noise - noise term inversely
proprtional to noise correlation time

Noise model 2: Power limited noise - noise term inversely proprtional to
square root of noise correlation time

@author: Lewis Dean
"""

import numpy as np
import matplotlib.pyplot as plt
from noise_spectral_density import (
    theoretical_autocorrelation, theoretical_power_spectral_density)

IMAGE_DIRECTORY = "plots/"

TAU_C_ARRAY = [0.1, 0.5, 1]
MODEL_NAMES = ["AOUP", "PLOUP"]
D_NE = 3

def plot_autocorrelation(time_delays, model_name, d_ne=D_NE,
                         tau_c_array=TAU_C_ARRAY):
    
    for tau_c in tau_c_array:
        
        if model_name == "AOUP":
            n = 1
        elif model_name == "PLOUP":
            n = 0.5
            
        autocorrelations = theoretical_autocorrelation(time_delays,
                                                       tau_c, n, d_ne)
        
        line_label = (r"$\tau_c$ = " + "{}".format(tau_c))
        plt.plot(time_delays, autocorrelations, label=line_label)

    plt.xlabel(r"$|\tau|$", fontsize=14)
    plt.ylabel(r"$\langle \, \zeta(t) \, \zeta(t + \tau) \, \rangle$",
               fontsize=14)
        
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne))
    
    plt.annotate(label, xy=(280, 178), xycoords='axes points',
        size=12, ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='w', edgecolor='#696969'))
    
    plt.xlim(0, np.max(time_delays))
    legend = plt.legend(loc=(0.55, 0.73), frameon=1)
    legend.get_frame().set_edgecolor('#696969')
    
    plt.savefig(IMAGE_DIRECTORY + 
                "analytic_noise_autocorrelation_plot_model_{}.svg".format(
                    model_name), dpi=600, format='svg')
    
    plt.show()
    plt.close()
    
    
def plot_power_spectral_density(f_linspace, model_name, d_ne=D_NE,
                                tau_c_array=TAU_C_ARRAY):
    
    for tau_c in tau_c_array:
        
        if model_name == "AOUP":
            n = 1
        elif model_name == "PLOUP":
            n = 0.5
    
        psd = theoretical_power_spectral_density(f_linspace, tau_c,
                                                 n, d_ne)
        line_label = (r"$\tau_c$ = " + "{}".format(tau_c))
        plt.plot(f_linspace, psd, label=line_label)
        
    plt.xlabel(r"$f$", fontsize=14)
    plt.ylabel(r"$S \, (\,f\,)$", fontsize=14)
        
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne))
    
    plt.annotate(label, xy=(280, 178), xycoords='axes points',
        size=12, ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='w', edgecolor='#696969'))
    
    legend = plt.legend(loc=(0.55, 0.73), frameon=1)
    legend.get_frame().set_edgecolor('#696969')
    
    plt.savefig(IMAGE_DIRECTORY +
                "analytic_noise_spectral_density_plot_model_{}.svg".format(
                    model_name), dpi=600, format='svg')
            
    plt.show()
    plt.close()
    
    
def main(model_names=MODEL_NAMES):
    
    time_delay_linspace = np.linspace(0, 10, 100)
    f_linspace = np.linspace(0, 10, 100)
    
    for model_name in model_names: 
            
            plot_autocorrelation(time_delay_linspace, model_name)
            
            plot_power_spectral_density(f_linspace, model_name)
            
    return None
        

if __name__ == "__main__":
    main()
