# -*- coding: utf-8 -*-
# cython: language_level=3
"""

Does this process involve a stiff SDE? Is there disparate
time scales for relaxation to thermal equilibrium and
the time between feedback updates?

Only free energy is a useful power output here as the negative
trap work is dissipated as heat to the colloidal medium. Hence,
the zero work feedback scheme will be used here, with alpha=2.

Initial coloured noise drawn from equilibrium distribution but
burn in period still needed as only the equilbirium distribution is known
to draw x_init, not the distribution for the NESS of the engine.

@author: Lewis Dean
"""

import numpy as np
cimport numpy as cnp
from scipy.stats import bootstrap
import model_constants as constants

cnp.import_array()
doubles = np.float64
ctypedef cnp.double_t doubles_t


cpdef return_empty(shape):
    
    return np.empty(shape, doubles)


cpdef double drift_term(double x, double lmb, float delta_g):
    
    return -(x-lmb) - delta_g


cpdef double equilibrium_diffusion_term(double dt=constants.DELTA_T):
    
    return np.sqrt(2*dt)


cpdef double noise_drift_term(double zeta, float tau_c):
    
    return -zeta / tau_c


cpdef double noise_diffusion_term(float tau_c, float d_ne,
                                  double dt=constants.DELTA_T,
                                  int model=constants.NOISE_MODEL):
    
    if model == 1:
        return np.sqrt(2*d_ne*dt) / tau_c
    elif model == 2:
        return np.sqrt(2*d_ne*dt/tau_c)


cpdef float wiener_increment():
    
    return np.random.normal(0, 1)


cpdef float generate_initial_active_noise(
    float tau_c, float d_ne,
    int model=constants.NOISE_MODEL,
    int equilibrium_noise=constants.EQUILBRIUM_NOISE_INITIALISATION):
    
    if not equilibrium_noise:
        return 0
    
    cdef float noise_std, initial_noise

    if model == 1:
        noise_std = np.sqrt(d_ne / tau_c)
    elif model == 2:
        noise_std = np.sqrt(d_ne)
    
    initial_noise = np.random.normal(0, noise_std)
    
    return initial_noise


cpdef double delta_zeta(double zeta, float tau_c, float d_ne, double dw,
                        double dt=constants.DELTA_T):
    
        return noise_drift_term(zeta, tau_c)*dt + noise_diffusion_term(
            tau_c, d_ne)*dw
    

cpdef double zeta_heun_method(double zeta, float tau_c, float d_ne,
                              double dt=constants.DELTA_T,
                              int model=constants.NOISE_MODEL):
    
       cdef double dw_zeta, zeta_estimate
    
       dw_zeta = wiener_increment()
       zeta_estimate = zeta + delta_zeta(zeta, tau_c, d_ne, dw_zeta)
            
       zeta = (zeta + 0.5*dt*(noise_drift_term(zeta, tau_c) + 
               noise_drift_term(zeta_estimate, tau_c)) + 
               noise_diffusion_term(tau_c, d_ne)*dw_zeta)
            
       return zeta
     

cpdef double delta_x(double x, double lmb, float delta_g,
                     double zeta, double dw_x, double dt=constants.DELTA_T):
    
    return (drift_term(x, lmb, delta_g) + zeta)*dt + equilibrium_diffusion_term() * dw_x


cpdef cnp.ndarray[cnp.double_t, ndim=1] x_euler_method(float tau_c, float d_ne, float gain,
                                                       float delta_g, int steps=constants.N,
                                                       double freq_measure=constants.MEASURING_FREQUENCY,
                                                       double dt=constants.DELTA_T,
                                                       float threshold=constants.THRESHOLD,
                                                       float offset=constants.OFFSET,
                                                       float transient_fraction=constants.TRANSIENT_FRACTION):
    
    cdef int n
    cdef int transient_steps = round(steps*transient_fraction)
    cdef double time, zeta, x, lmb, dw_x, ratchet_time
    cdef cnp.ndarray[cnp.double_t, ndim=1] ratchet_time_array = np.array([])
    
    time = 0
    zeta = generate_initial_active_noise(tau_c, d_ne)
    x = - delta_g
    lmb = 0
    ratchet_time = 0
    
    for n in range(steps):
        
        zeta = zeta_heun_method(zeta, tau_c, d_ne)           
            
        dw_x = wiener_increment()
        x = x + delta_x(x, lmb, delta_g, zeta, dw_x)
        
        time = (n+1)*dt
        ratchet_time = ratchet_time + dt
        
        if ((time*freq_measure).is_integer() and 
            (x - lmb) > threshold):
            
            lmb = lmb + gain*(x - lmb) + offset
            
            #Maybe causing the plotting error at t=10s
            if n+1 > transient_steps:
                ratchet_time_array = np.append(ratchet_time_array, ratchet_time)
            ratchet_time = 0
            
    if len(ratchet_time_array) == 0:
        print("No ratchet events for tau_c = {}".format(tau_c))
    
    return ratchet_time_array


cpdef double standard_error_on_mean(
    cnp.ndarray[cnp.double_t, ndim=1] ensemble):
   
    cdef double i
    cdef bint all_zeros = 1
    
    for i in ensemble:
        if i != 0:
            all_zeros = 0
            break
    if all_zeros:
        return 0
    
    cdef cnp.ndarray[cnp.double_t, ndim=2] ensemble_samples = (
        np.reshape(ensemble, (1, len(ensemble))))

    mean_distribution = bootstrap(ensemble_samples, np.mean,
                                  n_resamples=1000)
    
    return mean_distribution.standard_error


cpdef cnp.ndarray[cnp.double_t, ndim=1] mean_first_passage_time(
    float tau_c, float d_ne, float gain,
    float delta_g, int steps=constants.N,
    int samples=constants.SAMPLE_PATHS):
   
    cdef int j
    cdef cnp.ndarray[cnp.double_t, ndim=1] ratchet_time_ensemble
    cdef cnp.ndarray[cnp.double_t, ndim=1] mean_times = return_empty(samples)
    
    for j in range(samples):
        ratchet_time_ensemble = x_euler_method(tau_c, d_ne, gain, delta_g) 
        mean_times[j] = np.mean(ratchet_time_ensemble)
    # print(mean_times)

    cdef cnp.ndarray[cnp.double_t, ndim=1] mean_ratchet_time_results = (
        return_empty(2))
    mean_ratchet_time_results[0] = np.mean(mean_times)
    mean_ratchet_time_results[1] = standard_error_on_mean(mean_times)
    return mean_ratchet_time_results

