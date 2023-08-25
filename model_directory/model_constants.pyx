# -*- coding: utf-8 -*-
# cython: language_level=3
"""
Noise model 1: Active Ornstein-Uhlenbeck noise - noise term inversely
proportional to noise correlation time

Noise model 2: Power limited noise - noise term inversely proportional to
square root of noise correlation time

@author: Lewis Dean
"""

NOISE_MODEL = 2
EQUILIBRIUM_NOISE_INITIALISATION = 1

SAMPLE_PATHS = 1000
N = 250_000 # Number of time steps
PROTOCOL_TIME = 25
DELTA_T = PROTOCOL_TIME / N

MEASURING_FREQUENCY = 1/DELTA_T # Continuous sampling limit (all discrete timesteps)
THRESHOLD = 0
OFFSET = 0

TRANSIENT_FRACTION = 0.6