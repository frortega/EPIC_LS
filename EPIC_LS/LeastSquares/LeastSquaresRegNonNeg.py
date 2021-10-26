# -*- coding: utf-8 -*-
"""
by
Prof. Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

October 2021

Implements the solution of the General Least Squares Problem with Non Negativity 
constraints on all model parameters.

"""

import numpy as NP
from .LeastSquaresNonNeg import least_squares_non_neg

######
def least_squares_reg_weights_non_neg(G, d, Wx, H, ho, Wh):
    """
    Solves the general non negative least squares problem with weight matrices for both 
    misfit and regularization terms:

        Min_m ||Wx*(G*m-d)||_2^2 + ||Wh*(H*m-ho)||_2^2
            s.t.: m >= 0

    The problem is rewritten as an equivalent simple linear least squares problem:

        Min_m ||Fm-D||_2^2
            s.t.: m >= 0

    with adequate  F matrix and D vector.

    Returns a dictionary with keys 'm', 'misfit_norm', 'm_norm' and 'reg_norm' whose 
    values are the estimated model m, the norm of the misfit ||Wx*(G*m-d)||_2, the 
    norm of m ||m||_2 and the norm of the misfit of the regularized quantity ||H*m-ho||_2.

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

    m = least_squares_non_neg(F, D)['m']
    sol = {}
    sol['m'] = m
    # compute misfit, m and reg norms
    sol['misfit_norm'] = NP.linalg.norm(WxG.dot(m.flatten()) - Wxd.flatten())
    sol['m_norm'] = NP.linalg.norm(m.flatten())
    sol['reg_norm'] = NP.linalg.norm(H.dot(m.flatten()) - ho.flatten())

    return sol

def least_squares_reg_cov_non_neg(G, d, Cx, H, ho, Ch):
    """
    Solves the general non negative least squares problem with weight matrices for both 
    misfit and regularization terms:

        Min_m ||Wx*(G*m-d)||_2^2 +  ||Wh*(H*m-ho)||_2^2
            s.t.: m >= 0

    where Wx.T.dot(Wx) = inv(Cx) and Wh.T.dot(Wh) = inv(Ch)


    The problem is rewritten as an equivalent simple linear least squares problem:

        Min_m ||Fm-D||_2^2
            s.t.: m >= 0

    with adequate  F matrix and D vector.

    Returns a dictionary with keys 'm', 'misfit_norm', 'm_norm' and 'reg_norm' whose 
    values are the estimated model m, the norm of the misfit ||Wx*(G*m-d)||_2, the 
    norm of m ||m||_2 and the norm of the misfit of the regularized quantity ||H*m-ho||_2.    
    The obtained solution is a Maximum Likelihood solution.

    IMPORTANT: For this function to work,  all variables must be 2D arrays, i.e. vectors
    must be a numpy array with shape (N,1) where N is the number of elements.

    """

    # need to compute weight matrices using the cholesky decomposition of the covariance
    # matrices
    # for the misfit covariance matrix
    Ndata, Ndata2 = Cx.shape
    inv_Cx = NP.linalg.inv(Cx)
    Wx = NP.linalg.cholesky(inv_Cx).T  # as python calculates A.dot(A.T) = inv_Cx
    # for the prior information covariance matrix
    Nh, Nh2 = Ch.shape
    inv_Ch = NP.linalg.inv(Ch)
    Wh = NP.linalg.cholesky(inv_Ch).T  # as python calculates A.dot(A.T) = inv_Cx

    # compute and return the solution.
    return least_squares_reg_weights(G, d, Wx, H, ho, Wh)
