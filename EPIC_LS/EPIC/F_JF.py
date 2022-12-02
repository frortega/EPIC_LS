# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Calculation of EPIC residual and its Jacobian 

2022-12-02: - Adds optional minimum norm of diag(Wh) regularization to the calculation 
            of the EPIC.
            - Changes objective function to use target variances as the standard
            deviation of the EPIC in the nonlinear least squares problem.
"""
import numpy as NP
from scipy.linalg import inv

sigma_weight_default =  NP.exp(NP.finfo(float).precision/3)

### calculate function F (residuals) of the EPIC 
def calc_F(X, P, H, TargetVar, V = None, EPIC_bool = None, regularize = None):
    """
    Calculates function F (residuals) of the EPIC

    :param X: unknowns of the function, a 1D numpy array (the betas).
    :param P: precision matrix of the unregularized problem (2D numpy array)
    :param H: regularization operator (2D numpy array)
    :param TargetVar: 1D numpy array with target a posteriori variances
    :param V: if not None, must be a 2D numpy array that can be used to specify a linear 
              relationship between the different variables being search (X).
    :param EPIC_bool: A boolean numpy 1D array indicating which coefficients of m are 
               subject to the EPIC. If EPIC_bool[i] is True then m[i] is subject to EPIC.
               If var_m is the vector with the diagonal elements of the posterior 
               covariance matrix of model parameters (Cm), then, the EPIC is written as:
                    var_m[EPIC_bool] = target_sigmas**2
               CAUTION must be taked when defining EPIC_bool and target_sigmas as 
               the length and order of var_m[EPIC_bool] and target_sigmas**2  must match.
    :param regularize: if None, the EPIC condition is solved through an unregularized 
                       nonlinear least squares inversion. If a dictionary, can be an 
                       empty dictionary, or a dictionary defining 'sigma_weights', the
                       standard deviation of the minimum norm prior constraint on the
                       regularization weight. If the dictionary does not have the 
                       'sigma_weights' key, the default value is used
                       (default : NP.exp(NP.finfo(float).precision/3)).
    :return: Numpy array with function F evaluated in X.

    """

    if V is None:
        beta = X
    else:
        beta = V.dot(X)
    # assemble the inverse of Ch
    invCh = NP.diag(NP.exp(beta))
    # compute the EPIC residual vector
    A = P + H.T.dot(invCh.dot(H))
    invA = inv(A)
    # Assemble F
    if EPIC_bool is None:
        F = NP.diag(invA) - TargetVar
    else:
        F = NP.diag(invA)[EPIC_bool] - TargetVar

    F = F / TargetVar

    # extend F if using regularization to compute EPIC
    # Note that here the EPIC will be approximately met.
    if regularize is not None:
        # The standard deviation of the minimum norm of the weights Wh
        if 'sigma_weight' not in regularize.keys():
            sigma_weight = sigma_weight_default
        else:
            sigma_weight = regularize['sigma_weight']
            
        # damping on Wh, make them closer to null values
        F2 = NP.exp(beta/2) / sigma_weight
        
        F = NP.hstack((F, F2))    
        
    return F


### calculate the Jacobian JF of the functions of residuals F
def calc_JF(X, P, H, TargetVar, V = None, EPIC_bool = None, regularize = None):
    """
    Calculates the Jacobian JF of the functions of residuals F

    TargetVar is not used here, but is needes as a dummy argument for the main code.
    See calc_F() for an explanation of the arguments.

    """
    if V is None:
        beta = X
    else:
        beta = V.dot(X)

    invCh = NP.diag(NP.exp(beta))
    A = P + H.T.dot(invCh.dot(H))
    invA = inv(A)
    # assemble JF
    # fill the derivatives with respect to each beta
    B = H.dot(invA)
    BB = B * B
    E = NP.diag( NP.exp(beta) )
    JF = NP.transpose( -1.0 * E.dot(BB) )

    if V is not None:
        JF = JF.dot(V)

    if EPIC_bool is not None:
        JF = JF[EPIC_bool, :]

    JF = NP.diag(1/TargetVar).dot(JF)

    # extend JF if using regularization to compute EPIC
    # Note that here the EPIC will be approximately met.
    if regularize is not None:
        if 'sigma_weight' not in regularize.keys():
            sigma_weight = sigma_weight_default
        else:
            sigma_weight = regularize['sigma_weight']

        # add the jacobian of the Wh damping
        JF2 = 0.5 * NP.diag(NP.exp(beta/2)) / sigma_weight
    
        JF = NP.vstack((JF, JF2))

    return JF


