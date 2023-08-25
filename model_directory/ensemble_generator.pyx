# -*- coding: utf-8 -*-
# cython: language_level=3
"""

Does this process involve a stiff SDE? Is there disparate
time scales for relaxation to thermal equilibrium and
the time between feedback updates?

Initial coloured noise drawn from equilibrium distribution but
burn in period still needed as only the equilbirium distribution is known
to draw x_init, not the distribution for the NESS of the engine.

To check bootstrapping is working correctly, should draw from a gaussian
distribution and try to recover the variance via bootstrapping.

@author: Lewis Dean
"""

import numpy as np
cimport numpy as cnp
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
    int equilibrium_noise=constants.EQUILIBRIUM_NOISE_INITIALISATION):
    
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
    '''
    Could zeta heun be what is producing the errors?
    '''
    
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


cpdef cnp.ndarray[cnp.double_t, ndim=2] x_euler_method(
    float tau_c, float d_ne, float gain,
    float delta_g, int steps=constants.N,
    double freq_measure=constants.MEASURING_FREQUENCY,
    double dt=constants.DELTA_T,
    float threshold=constants.THRESHOLD,
    float offset=constants.OFFSET,
    float transient_fraction=constants.TRANSIENT_FRACTION):
    
    cdef int n, i
    cdef bint transience_finished
    cdef double time, zeta, x, lmb, dw_x
    cdef int transient_steps = round(steps*transient_fraction)
    cdef cnp.ndarray[cnp.double_t, ndim=1] x_array = (
        return_empty(steps - transient_steps +1))
    cdef cnp.ndarray[cnp.double_t, ndim=1] lmb_array = (
        return_empty(steps-transient_steps+1))
    
    transience_finished = 0
    time = 0
    x = - delta_g
    lmb = 0
    zeta = generate_initial_active_noise(tau_c, d_ne)
    
    for n in range(steps):
        
        zeta = zeta_heun_method(zeta, tau_c, d_ne)
        
        if n < transient_steps:
                
            dw_x = wiener_increment()
            x = x + delta_x(x, lmb, delta_g, zeta, dw_x)
            
            time = (n+1)*dt
            
            if ((time*freq_measure).is_integer() and 
                (x - lmb) > threshold):
                lmb = lmb + gain*(x - lmb) + offset
            
        else:
            
            if not transience_finished:
                x_array[0] = x
                lmb_array[0] = lmb
                transience_finished = 1
    
            i = n - transient_steps
            x, lmb = x_array[i], lmb_array[i]
                
            dw_x = wiener_increment()
            x_array[i+1] = x + delta_x(x, lmb, delta_g, zeta, dw_x)
            
            time = (n+1)*dt
            
            if ((time*freq_measure).is_integer() and 
                (x - lmb) > threshold):
                lmb_array[i+1] = lmb + gain*(x - lmb) + offset
            else:
                lmb_array[i+1] = lmb
                
    return np.array([x_array, lmb_array])


cpdef cnp.ndarray[cnp.double_t, ndim=3] generate_trajectory_ensemble(float tau_c, float d_ne, float gain,
                                                                     float delta_g, int steps=constants.N,
                                                                     int samples=constants.SAMPLE_PATHS,
                                                                     float transient_fraction=constants.TRANSIENT_FRACTION):
   
    cdef int j
    cdef int transient_steps = round(steps*transient_fraction)
    cdef cnp.ndarray[cnp.double_t, ndim=3] trajectories = (
        return_empty((samples, 2, steps-transient_steps+1)))
    
    for j in range(samples):
        trajectories[j] = x_euler_method(tau_c, d_ne, gain, delta_g)
    
    return trajectories
