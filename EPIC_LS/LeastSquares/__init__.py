# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Modifications: 
October 2021 - Adds non negative least squares (unregularized)

"""
from .LeastSquares import least_squares, least_squares_cov, least_squares_weights
from .LeastSquaresReg import least_squares_reg_weights, least_squares_reg_cov
from .LeastSquaresNonNeg import least_squares_non_neg, least_squares_non_neg_weights,\
                                least_squares_non_neg_cov
