# -*- coding: utf-8 -*-
"""
by
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2021

Implements unregularized linear least squares inversion with non negativity constraints.
The NNLS (scipy.optimize) package is used.

"""
from scipy.optimize import nnls
import numpy as NP
######
def least_squares_non_neg(G, d):
    """
    Estimates the solution of G*m = d using non negative least squares.
    Uses scipy.optimize.nnls implementation of non negative least squares.

    Returns the estimated solution and the norm of the misfit residual.

    NOTE: All variables must be 2D numpy arrays, vectors are of shape (N,1)

    """
    m, ResNorm = nnls(G, d.squeeze())
    return {'m' : m, 'ResNorm' : ResNorm}

######
def least_squares_non_neg_weights(G, d, Wx):
    """
    Estimates the solution of Wx*G*m = Wx*d using non negative least squares.
    Uses scipy.optimize.nnls implementation of non negative least squares.

    Returns the estimated solution and the norm of the misfit residual (including weights.

    NOTE: All variables must be 2D numpy arrays, vectors are of shape (N,1)

    """
    return least_squares_non_neg(Wx.dot(G), Wx.dot(d))


######
def least_squares_non_neg_cov(G, d, Cx):
    """
    Estimates the solution of Wx*G*m = Wx*d using non negative least squares, where
    Wx is such that inv(Cx) = Wx.T.dot(Wx). Cx is the misfit error covariance matrix.

    Uses scipy.optimize.nnls implementation of non negative least squares.

    Returns the estimated solution and the norm of the misfit residual (including weights.

    NOTE: All variables must be 2D numpy arrays, vectors are of shape (N,1)

    """
    Ndata, Ndata2 = Cx.shape
    inv_Cx = NP.linalg.lstsq( Cx , NP.eye(Ndata) )[0]
    Wx = NP.linalg.cholesky( inv_Cx  ).T # as python calculates A.dot(A.T) = inv_Cx    
 
    return  least_squares_non_neg_weights(G, d, Wx)

