# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:37:10 2023

Perhaps, I should do a similar thing for particle positions, I should
plot a probability density of x as a function of elapsed time and check
at which point the pdf is approximately constant. The burn-in period is
the length of time required to satisfy such a condition. However, this
may never work if in a NESS - probability flux exists so mean x should 
reach a cyclic state.

Seeing how drastically the equilibrium initialisation influenced the 
active noise, using an appropriate transient period for the information
engine's position SDE seems essential.
Perhaps use 50 tau_r and a transient period of 50%. Then use 1,000,000
steps such that the discretisation error should be even smaller here.
Also continue to use 1000 sampled trajectories.

@author: lewis
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import model_constants as constants
from ensemble_generator import generate_initial_active_noise, zeta_heun_method

# ZETA_PDF_FILENAME = "equilibrium_zeta_pdfs_gain_2_model_1.csv"
ZETA_PDF_FILENAME = "transient_zeta_pdfs_gain_2_model_1.csv"
# ZETA_PDF_FILENAME = "equilibrium_zeta_pdfs_gain_2_model_2.csv"
# ZETA_PDF_FILENAME = "transient_zeta_pdfs_gain_2_model_2.csv"

IMAGE_DIRECTORY = "plots/"
DATA_DIRECTORY = "data_files/"

GENERATE_DATA = 0
NUMBER_OF_CORES_USED = -1

GAIN = 2
# Be careful changing these before plotting old data...
HISTOGRAM_BIN_NUM = 50
D_NE = 3
TAU_C = 10**4

PLOT_ZETA_PDFS = 1
TIME_BETWEEN_PLOTS = 10**-1

STD_TOLERANCE_FRACTION = 0.05


def identify_data_parameters(data_filename):
    
    initialisation = re.findall("[^_]*", data_filename)[0]
    if initialisation == "equilibrium":
        equilibrium_noise = 1
    elif initialisation == "transient":
        equilibrium_noise = 0
        
    parameters = re.findall(r'\d+(?:\.\d+)?', data_filename)
    feedback_gain = parameters[0]
    noise_model = parameters[1]
    
    return equilibrium_noise, float(feedback_gain), int(noise_model)


def calculate_noise_evolution(tau_c, d_ne, model, steps=constants.N):
        
    zeta_array = np.empty(steps+1)  
    zeta_array[0] = generate_initial_active_noise(tau_c, d_ne, model)
    for n in range(steps):
        zeta_array[n+1] = zeta_heun_method(zeta_array[n], tau_c, d_ne)
    return zeta_array


def noise_path_generator(thread, tau_c, d_ne, model):
    
    return  calculate_noise_evolution(tau_c, d_ne, model)
    
    
def generate_noise_ensemble(tau_c, d_ne, model=constants.NOISE_MODEL,
                            samples=constants.SAMPLE_PATHS,
                            n_cores=NUMBER_OF_CORES_USED):
    
    noise_trajectories = Parallel(n_jobs=n_cores,
                                  backend = "threading", verbose=1)(
                                  delayed(noise_path_generator)(
                                  thread, tau_c,
                                  d_ne, model) for thread in range(samples))

    return np.array(noise_trajectories)


def calculculate_noise_pdfs(noise_trajectories, steps=constants.N,
                            samples=constants.SAMPLE_PATHS,
                            num_of_bins=HISTOGRAM_BIN_NUM):
    
    zeta_bin_arrays = np.empty((steps+1, num_of_bins+1))
    zeta_count_arrays =  np.empty((steps+1, num_of_bins))
    
    for n in range(steps):
    
        zeta_count_arrays[n], zeta_bin_arrays[n] = (
            np.histogram(noise_trajectories[:, n], bins=num_of_bins))
        zeta_std = np.std(noise_trajectories[:, n], ddof=1)
                
    return zeta_bin_arrays, zeta_count_arrays, zeta_std


def save_noise_pdfs(zeta_bin_arrays, zeta_count_arrays, zeta_stds, gain=GAIN,
                    model=constants.NOISE_MODEL, steps=constants.N,
                    dt=constants.DELTA_T, num_of_bins=HISTOGRAM_BIN_NUM,
                    equilbrium_noise=constants.EQUILIBRIUM_NOISE_INITIALISATION):
    

    times = np.linspace(0, steps*dt, steps+1)
    
    data = {"Times": times}
    for i in range(num_of_bins):
        data[f"Noise bin {i+1}"] = zeta_bin_arrays[:, i]
        data[f"Noise count {i+1}"] = zeta_count_arrays[:, i]
    data["Noise std"] = zeta_stds
    
    df = pd.DataFrame(data)
    df.set_index('Times', inplace=True)
    
    if equilbrium_noise:
        file_name = "equilibrium_zeta_pdfs_gain_{0}_model_{1}.csv".format(
            gain, model)
    else:
        file_name = "transient_zeta_pdfs_gain_{0}_model_{1}.csv".format(
            gain, model)
    
    df.to_csv(DATA_DIRECTORY + file_name, sep=',')

    return None


def plot_zeta_pdfs(zeta_pdf_data, equilibrium_noise, model_name,
                   tau_c=TAU_C, d_ne=D_NE, dt=constants.DELTA_T,
                   total_time=constants.PROTOCOL_TIME,
                   time_between_plots=TIME_BETWEEN_PLOTS,
                   tolerance=STD_TOLERANCE_FRACTION):
    
        equilibriated = False
        
        times = zeta_pdf_data[:-1,0]
        all_bins  = zeta_pdf_data[:, 1:-1:2]
        all_counts = zeta_pdf_data[:, 2::2]
        all_stds = zeta_pdf_data[:, -1]
        
        if model_name == "AOUP":
            theory_noise_std = np.sqrt(d_ne/tau_c)
        if model_name == "PLOUP":
            theory_noise_std = np.sqrt(d_ne)
        
        step_interval = int(time_between_plots/dt)
        for index in range(0, len(times), step_interval):
            
            
            time = times[index]
            bins = all_bins[index]
            delta_bin = bins[-1] - bins[-2]
            bins = np.append(bins, bins[-1] + delta_bin)
            counts = all_counts[index]
            std = all_stds[index]
            
            if (theory_noise_std - std <= tolerance*theory_noise_std and
                not equilibriated):
                equilibriated = True
                print("Noise equilibriates at t = {}".format(time))
            
            fig = plt.figure()
            axes = fig.add_subplot(111)
        
            axes.set_title(r"Distribution of $t$ = " + "{:.4g} ".format(time) +
                      r"$\zeta$ values")
            axes.set_xlabel(r"$\zeta$")
            axes.set_ylabel(r"$\zeta$ counts")
            axes.set_ylim
            
            axes.axvline(theory_noise_std, c='r', linestyle="--")
            axes.axvline(-theory_noise_std, c='r', linestyle="--")
            
            axes.axvline(std, c='g', linestyle="--")
            axes.axvline(-std, c='g', linestyle="--")
            
            label = ("{}: ".format(model_name) + "\n" +
                     "{} initialisation ".format(
                         "Equilibrium" if equilibrium_noise==1 
                         else "Transient") + "\n" +
                     r"$\tau_c$ =" +" {:.3g}".format(tau_c) + "\n" +
                     r"$D_{ne}$ =" + " {:.3g}".format(d_ne))
            
            textbox = dict(boxstyle='round', edgecolor='k', facecolor='w',
                           alpha=0.75)
            
            axes.text(1.04, 0.75, label, transform=axes.transAxes,
                      fontsize=10, verticalalignment='bottom', bbox=textbox)
            
            axes.stairs(counts, bins, fill=True)
            
            total_time = 10
            if (time == total_time - time_between_plots and 
                equilibrium_noise == 0):
                plt.savefig(IMAGE_DIRECTORY + 
                            "{}_transient_zeta_distribution_after_burnin.png".format(
                                model_name), format="png", bbox_inches='tight',
                            dpi=600)
    
            plt.show()
            plt.close()


def main(tau_c=TAU_C, d_ne=D_NE, filename=ZETA_PDF_FILENAME):
    
    if GENERATE_DATA:
            
        noise_trajectories = generate_noise_ensemble(tau_c, d_ne)
        zeta_bin_arrays, zeta_count_arrays, zeta_stds = calculculate_noise_pdfs(
            noise_trajectories)
        save_noise_pdfs(zeta_bin_arrays, zeta_count_arrays, zeta_stds)
        
    if PLOT_ZETA_PDFS:
        
        equilibrium_noise, gain, model = identify_data_parameters(filename)
        if model == 1:
            model_name = "AOUP"
        elif model == 2:
            model_name = "PLOUP"
        
        zeta_pdf_data = np.genfromtxt(DATA_DIRECTORY + filename,
                                      delimiter=',', skip_header=1)
        
        plot_zeta_pdfs(zeta_pdf_data, equilibrium_noise, model_name)
    
    return None


if __name__ == "__main__":
    main()
