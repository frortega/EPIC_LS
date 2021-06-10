# -*- coding: utf-8 -*-
__doc__ = """
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Calculation of bounds for betas to prevent floating point rounding errors.

"""
import numpy as NP

# get machine precision for float numbers
eps = NP.finfo(float).eps

def compute_bounds(k_center = 0, distance = 2):
    """
    compute the largest k, so that NP.exp( k_center - k ) + NP.exp( k_center + k )
    can be well represented with machine floating precission.
    :return: k
    """
    k_test = 0
    for i in range(1, 1000000):
        k_test += 0.01
        sum = NP.exp(k_center - k_test) +  NP.exp(k_center + k_test) \
              - NP.exp(k_center + k_test)
        if NP.abs(sum) < eps:
            break

    k_test = k_test - distance # keep some distance from such limit.

    return (k_center - k_test, k_center + k_test )



