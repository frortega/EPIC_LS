# -*- coding: utf-8 -*-
"""
by
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Implements unregularized linear least squares inversion.

"""
import numpy as NP


######
def least_squares(G, d):
    """
    Estimates the unknown parameter vector m in G*m = d + eta using simple linear least
    squares. Here eta is the error of the misfit (including data and model prediction
    errors), assumed to be i.i.d. with zero mean and a variance with unitary value.

    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.

    The solution of this problem is a Maximum Likelihood solution if misfit errors are
    i.i.d. with zero mean and a variance with unitary value.

    """
    Ndata, Npar = G.shape
    # compute inverse of a posteriori covariance matrix
    invCm = G.T.dot(G)
    # compute a posteriori covariance matrix
    Cm = NP.linalg.inv(invCm)
    # compute the linear least squares solution
    m = Cm.dot( G.T.dot(d) )
    return {'m': m, 'Cm': Cm}


######
def least_squares_weights(G, d, Wx):
    """
    Estimates the unknown parameter vector m in Wx*G*m = Wx*d + eta using simple linear
    least squares. eta is the error of the misfit (including data and model prediction
    errors), assumed to be i.i.d. with zero mean and a variance with unitary value.

    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.

    The solution of this problem is a Maximum Likelihood solution if Wx is such that
    Wx.T.dot(Wx) = inv(Cx) where Cx = Cd + Cp is the covariance matrix of the misfit, and
    Cd, Cp are the covariance matrices of the data and model prediction, respectively.

    """
    return least_squares(Wx.dot(G), Wx.dot(d))


######
def least_squares_cov(G, d, Cx):
    """
    Estimates the unknown parameter vector m in Wx*G*m = Wx*d + eta using simple linear
    least squares. eta is the error of the misfit (including data and model prediction
    errors), assumed to be i.i.d. with zero mean and a variance with unitary value, and
    Wx is such that inv(Cx) = Wx.T.dot(Wx), where Cx is the a priori covariance matrix of
    the misfit (Cx = Cd + Cp, where Cd, Cp are the covariance matrices of the data and
    model prediction, respectively). Wx is calculated as the Cholesky decomposition of Cx.

    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.

    The solution of this problem is a Maximum Likelihood solution.

    """
    # calculate Wx
    Ndata, Ndata2 = Cx.shape
    inv_Cx = NP.linalg.lstsq( Cx , NP.eye(Ndata) )[0]
    Wx = NP.linalg.cholesky( inv_Cx  ).T # as python calculates A.dot(A.T) = inv_Cx    

    # return the solution of the problem.
    return least_squares_weights(G, d, Wx)

