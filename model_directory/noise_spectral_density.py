# -*- coding: utf-8 -*-
"""
F_NE of 10**4 is the white noise limit. F_NE of 10**-4 is
very correlated noise. Opposite for TAU_C.

Data looks AWFUL in the white noise limit - no correlation...
To make smoother plots, use a higher correlation time, e.g. tau_c = 10
When using this tau, the protocol time is not enough!
So resort to tau_c = 1

When using a long protocol time, the correlations tend to zero, therefore,
even for autocorrelations from 10,000 trajectory ensembles, some values dip
below zero, then you get a negative logarithm and the plot breaks! So try
a smaller protocol time which aligns with the tau_c chosen. For example,
I will use a protocol time of 3 for tau_c = 1.

@author: Lewis Dean
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import model_constants as constants
from ensemble_generator import generate_initial_active_noise, zeta_heun_method


IMPORT_FILES = ["autocorrelation_data_tau_c_1_model_1.csv"]

IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"

GENERATE_DATA = 0
NUMBER_OF_CORES_USED = -1
TAU_C = 1
# Be careful changing this before plotting old data...
D_NE = 3

PLOT_AUTOCORRELATION = 1
PLOT_PSD = 1
PSD_XLIM = 2
SAVE_PLOTS = 0


def identify_data_models(data_filename):

    parameters = re.findall(r'\d+(?:\.\d+)?', data_filename)
    tau_c = parameters[0]
    model_type = parameters[1]

    return float(tau_c), int(model_type)


def generate_time_delays(dt=constants.DELTA_T, steps=constants.N):
    
    return np.linspace(0, steps*dt, steps+1)


def calculate_noise_evolution(tau_c, d_ne, model, steps=constants.N):
        
    zeta_array = np.empty(steps+1)  
    zeta_array[0] = generate_initial_active_noise(tau_c, d_ne, model)
    for n in range(steps):
        zeta_array[n+1] = zeta_heun_method(zeta_array[n], tau_c,
                                           d_ne)
    
    return zeta_array


def noise_path_generator(thread, tau_c, d_ne, model):
        
    return  calculate_noise_evolution(tau_c, d_ne, model)
    
    
def generate_noise_ensemble(tau_c, d_ne, model,
                            samples=constants.SAMPLE_PATHS,
                            n_cores=NUMBER_OF_CORES_USED):
    
    noise_trajectories = Parallel(n_jobs=n_cores,
                                  backend = "threading", verbose=1)(
                                  delayed(noise_path_generator)(
                                  thread, tau_c,
                                  d_ne, model) for thread in range(samples))

    return np.array(noise_trajectories)


def calculculate_autocorrelations(noise_trajectories,
                                 samples=constants.SAMPLE_PATHS,
                                 steps=constants.N):
    
    zeta_products = np.empty((samples, steps + 1))
    for sample_index, zeta_array in enumerate(noise_trajectories):
        for zeta_index, zeta in enumerate(zeta_array):

                zeta_products[sample_index, zeta_index] = zeta * zeta_array[0]
    
    zeta_autocorrelations = np.mean(zeta_products, axis=0)
    
    return zeta_autocorrelations


def theoretical_autocorrelation(time_delay, tau_c, n, d_ne):
    
    return d_ne*tau_c**(1-2*n)*np.exp(-time_delay/tau_c)


def plot_autocorrelation(file_time_delays, file_autocorrelations,
                         model_names, tau_c_list ,d_ne=D_NE,
                         import_files=IMPORT_FILES):
    
    for i in range(len(import_files)):
        
        time_delays = file_time_delays[i]
        autocorrelations = file_autocorrelations[i]
        model_name = model_names[i]
        tau_c = tau_c_list[i]
        
        if model_name == "AOUP":
            n = 1
        elif model_name == "PLOUP":
            n = 0.5
        
        t_delay_linspace = np.linspace(0, time_delays[-1], 100)
        fitted_autocorrelation = theoretical_autocorrelation(t_delay_linspace,
                                                             tau_c, n, d_ne)
        
        plt.title((r"$\langle \zeta(t)\zeta(t + \tau) \rangle$ as a " +
                   r"function of $\tau$"), fontsize=14)
        plt.xlabel(r"$|\tau|$", fontsize=14)
        plt.ylabel(r"$\langle \zeta(t)\zeta(t + \tau) \rangle$", fontsize=14)
        
        line_label = (r"$\tau_c$ = " + "{}".format(tau_c))
        
        plt.plot(time_delays, autocorrelations, label=line_label)
        plt.plot(t_delay_linspace, fitted_autocorrelation,
                 label=r"$\tau_c$" + " {} autocorrelation fit".format(
                     tau_c))
        
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne))
    
    plt.annotate(label, xy=(280, 175), xycoords='axes points',
        size=12, ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='w', edgecolor='#696969'))
    
    plt.xlim(0, np.max(np.array(file_time_delays)[:, -1]))
    legend = plt.legend(loc=(0.3, 0.6), frameon=1)
    legend.get_frame().set_edgecolor('#696969')
    
    if SAVE_PLOTS:
        plt.savefig(IMAGE_DIRECTORY + 
                    "noise_autocorrelation_plot_model_{}.svg".format(
                        model_name), dpi=600, format='svg')
    
    plt.show()
    plt.close()
    

def theoretical_power_spectral_density(f, tau_c, n, d_ne):
    
    part_1 = d_ne*tau_c**(1-2*n)
    part_2 = 1/(tau_c**2) + (2*np.pi*f*tau_c)**2
    
    return part_1/part_2
        

def plot_power_spectral_density(file_autocorrelations, delta_t,
                                model_names, tau_c_list,
                                d_ne=D_NE, import_files=IMPORT_FILES,
                                psd_xlim=PSD_XLIM):
    
    for i in range(len(import_files)):
        
        autocorrelations = file_autocorrelations[i]
        model_name = model_names[i]
        tau_c = tau_c_list[i]
        
        freq = np.fft.fftfreq(len(autocorrelations), delta_t)
        f_transform = np.fft.fft(autocorrelations, norm="ortho")
        psd = np.abs(f_transform)
        
        # Needed to remove zero frequency line
        freq_shifted = np.fft.fftshift(freq)
        psd_shifted = np.fft.fftshift(psd)
        
        if model_name == "AOUP":
            n = 1
        elif model_name == "PLOUP": 
            n = 0.5
        
        f_linspace = np.linspace(0, psd_xlim)
        fitted_psd = theoretical_power_spectral_density(f_linspace,
                                                        tau_c, n, d_ne)
        # normalisation_constant = (
        #     len(autocorrelations)*delta_t/(d_ne*tau_c**(2*(1-n))*1 - np.exp(-2*t_burn/tau_c)))
        
        normalisation_constant = psd[0]/fitted_psd[0]
        print(normalisation_constant)
        fitted_psd = normalisation_constant * fitted_psd
        
        plt.title("Nonequilibrium noise power spectral density")
        plt.xlabel(r"$f$", fontsize=14)
        plt.ylabel("Power spectral density", fontsize=14)
        line_label = (r"$\tau_c$ = " + "{}".format(tau_c))
        plt.plot(freq_shifted, psd_shifted, label=line_label)
        plt.plot(f_linspace, fitted_psd, linestyle='--',
                 label=r"$\tau_c$" + " = {} Fitted PSD".format(tau_c))
        
    label = ("{}: ".format(model_name) + "\n" + 
             r"$D_{ne}$ = " + "{}".format(d_ne))
    
    plt.annotate(label, xy=(268, 157), xycoords='axes points',
        size=12, ha='left', va='bottom',
        bbox=dict(boxstyle='round', fc='w', edgecolor='#696969'))
    
    plt.xlim(0, psd_xlim)
    legend = plt.legend(loc=(0.3, 0.6), frameon=1)
    legend.get_frame().set_edgecolor('#696969')
    
    if SAVE_PLOTS:
        plt.savefig(IMAGE_DIRECTORY +
                    "noise_spectral_density_plot_model_{}.svg".format(
                        model_name), dpi=600, format='svg')
            
    plt.show()
    plt.close()


def main(tau_c=TAU_C, d_ne=D_NE,
         delta_t=constants.DELTA_T, model=constants.NOISE_MODEL,
         import_files=IMPORT_FILES):
    
    if GENERATE_DATA:
        
        time_delays = generate_time_delays()
        noise_trajectories = generate_noise_ensemble(tau_c, d_ne, model)
        
        autocorrelations = calculculate_autocorrelations(
            noise_trajectories)
        
        file = np.zeros((len(autocorrelations), 2))
        file[:,0] = time_delays
        file[:,1] = autocorrelations
        headers = ["time_delay", "autocorrelation"]
        df = pd.DataFrame(file, columns=headers)
        df.to_csv((DATA_DIRECTORY +
                  "autocorrelation_data_tau_c_{0}_model_{1}.csv".format(
                      tau_c, model)), index=0, header=1, sep=',')
            
    else:
    
        file_time_delays = []
        file_autocorrelations = []
        tau_c_list = []
        model_names = []
        
        for file in import_files:
                        
            data = np.genfromtxt(DATA_DIRECTORY + file, delimiter=',',
                                 skip_header=1)
            
            file_time_delays += [data[:,0]]
            file_autocorrelations += [data[:,1]]
            
            tau_c, model = identify_data_models(file)
            tau_c_list += [tau_c]
            if model == 1:
                model_names += ["AOUP"]
                n = 1
            elif model == 2:
                model_names += ["PLOUP"]
                n = 0.5
                
            time_delays = file_time_delays[-1]
            autocorrelations = file_autocorrelations[-1]
            tau_c_autocorrelation = autocorrelations[np.argmin(
                abs(time_delays - tau_c))]
            print("e-folded autocorrelation for noise of tau_c = {0} is {1:.5f}.".format(
                tau_c, autocorrelations[0]/np.exp(1)))
            print("Autocorrelation at tau_c = {0} is {1:.5f}.".format(
                tau_c, tau_c_autocorrelation))
            print("Theoretical autocorrelation at tau_c = {0} is {1:.5f}.\n ".format(
                tau_c, theoretical_autocorrelation(tau_c, tau_c, n, d_ne)))

        if PLOT_AUTOCORRELATION:
                
            plot_autocorrelation(file_time_delays, file_autocorrelations,
                                 model_names, tau_c_list)
        
        if PLOT_PSD:
            
            plot_power_spectral_density(file_autocorrelations, delta_t,
                                        model_names, tau_c_list)
            
    return None


if __name__ == '__main__':
    main()
