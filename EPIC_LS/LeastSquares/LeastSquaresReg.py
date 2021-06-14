# -*- coding: utf-8 -*-
"""
by
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Implements the solution of the General Least Squares Problem.

"""

import numpy as NP
from .LeastSquares import least_squares

######
def least_squares_reg_weights(G, d, Wx, H, ho, Wh):
    """
    Solves the general least squares problem with weight matrices for both misfit and
    regularization terms:

        Min_m ||Wx*(G*m-d)||_2^2 + ||Wh*(H*m-ho)||_2^2

    The problem is rewritten as an equivalent simple linear least squares problem:

        Min_m ||Fm-D||_2^2

    with adequate  F matrix and D vector.

    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.

    The obtained solution is a Maximum Likelihood solution if  Wx.T.dot(Wx) = inv(Cx)
    and Wh.T.dot(Wh) = inv(Ch), where Cx and Ch are the misfit and prior information
    covariance matrices, respectively.

    IMPORTANT: For this function to work,  all variables must be 2D arrays, i.e. vectors
    must be a numpy array with shape (N,1) where N is the number of elements.

    """
    # make sure that d and ho are column vectors
    d = d.reshape((len(d),1))
    ho = ho.reshape((len(ho),1))
    # work on the theory equations
    WxG = Wx.dot(G)
    Wxd = Wx.dot(d)
    # work on the regularization equations
    WhH =  Wh.dot(H)
    Whho = Wh.dot(ho)

    # Assemble the equivalent simple linear least square problem.
    F = NP.vstack([WxG, WhH])
    D = NP.vstack([Wxd, Whho])

    sol = least_squares(F, D)

    return sol

def least_squares_reg_cov(G, d, Cx, H, ho, Ch):
    """
    Solves the general least squares problem with weight matrices for both misfit and
    regularization terms:

        Min_m ||Wx*(G*m-d)||_2^2 +  ||Wh*(H*m-ho)||_2^2

    where Wx.T.dot(Wx) = inv(Cx) and Wh.T.dot(Wh) = inv(Ch)


    The problem is rewritten as an equivalent simple linear least squares problem:

        Min_m ||Fm-D||_2^2

    with adequate  F matrix and D vector.

    Returns a dictionary with keys 'm' and 'Cm' whose values are the estimated model m and
    its a posteriori covariance matrix Cm, respectively.

    The obtained solution is a Maximum Likelihood solution.

    IMPORTANT: For this function to work,  all variables must be 2D arrays, i.e. vectors
    must be a numpy array with shape (N,1) where N is the number of elements.

    """

    # need to compute weight matrices using the cholesky decomposition of the covariance
    # matrices
    # for the misfit covariance matrix
    Ndata, Ndata2 = Cx.shape
    inv_Cx = NP.linalg.lstsq(Cx, NP.eye(Ndata))[0]
    Wx = NP.linalg.cholesky(inv_Cx).T  # as python calculates A.dot(A.T) = inv_Cx
    # for the prior information covariance matrix
    Nh, Nh2 = Ch.shape
    inv_Ch = NP.linalg.lstsq(Ch, NP.eye(Nh))[0]
    Wh = NP.linalg.cholesky(inv_Ch).T  # as python calculates A.dot(A.T) = inv_Cx

    # compute and return the solution.
    return least_squares_reg_weights(G, d, Wx, H, ho, Wh)
