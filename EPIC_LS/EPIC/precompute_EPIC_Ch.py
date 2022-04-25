# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2021

Modules to precompute EPIC Ch for a Least Squares Problem, and store it into a file.

TODO: Need to control verbosity in the main loop of precompute_EPIC_Ch

"""
import numpy as NP
import scipy as SP
import os
from .calc_EPIC_Ch import calc_EPIC_Ch
import pickle

def precompute_EPIC_Ch(G, Cx, H, target_sigmas, X0 = None, V = None,
            LSQpar={}, homogeneous_step = True,
            beta_shift_k=0, beta_distance=2,
            EPIC_bool = None):
    """

    :param G: Design matrix with Green's functions of the problem (Nd x Nm)
    :param Cx: Misfit covariance matrix (Nd x Nd)
    :param H: Regularization operator (matrix) (Nh x Nm)
    :param target_sigmas: a list containing vectors, numpy arrays with shape (Nm,1),
        with the target errors (standard deviations) of the model parameters.
        The idea behind EPIC is that those errors are constant for each spatially
        discretized value (e.g., for strike slip, and dip slip), but those can be
        different (i.e., target variance for strike slip may be different from that
        for dip slip). target_sigmas can also be a list of floats, in which is assumed
        that the float number is the target sigma for all the parameters.

    :param X0: initial model for a priori variances for the regularization (Nh x 1)
                if X0 = None, X0 is taken equal to 0 for all elements.
                X0 = -NP.log(Ch0)
    :param LSQpar: a dictionary with several parameters that control convergence of
                nonlinear optimization algorithm used to solve the EPIC condition problem.
    :param V: matrix accounting for a linear variable change, x = V.dot(y) in which
             we search values for y instead of x.
    :param beta_shift_k & beta_distance: see docstring of beta_bounds.compute_bounds
    :param homogeneous_step: if True does first an homogeneous step to find a preliminary
                            initial guess of Ch.
    :param LSQpar: must be a dictionary containing the convergence parameters for:
        (1) Homogeneous step search (with default values of):
            - LSQpar['TolX1'] = 1e-6
            - LSQpar['TolFun1'] = 1e-6
            - LSQpar['TolG1'] = 1E-6
        (2) search for heterogenous Ch (with default values of):
            - LSQpar['TolX2'] = 1e-8
            - LSQpar['TolFun2'] = 1e-8
            - LSQpar['TolG2'] = 1E-10
        (3) Solver, loss function type  and verbose level
            - LSQpar['method'] = 'trf'
            - LSQpar['loss'] = 'linear'
            - LSQpar['verbose'] = 2

        see scipy.optimize.least_squares help for further information. Here, TolX?, TolF?
        and TolG? refer to tolerances defined for convergence criteria on model, objective
        function and gradient variations, respectively.

    :return: a list in which each item is a dictionary with the estimated vector of a
    priori variances Ch and status information on the results of the nonlinear
    optimization used to calculate the EPIC condition. The order of the list is the same
    order in which
    """

    NumTargetSigmas = len(target_sigmas)

    # initialize container for solutions.
    ChSol = []

    # prepare variables needed for calculation of Ch
    Ndata, Npar = G.shape
    Nh, Npar2 = H.shape
    # check that G and H apply to the same number of parameters
    if Npar != Npar2:
        raise ValueError('G and H must have the same number of columns!...')

    # precision matrix of the unregularized problem
    inv_Cx = NP.linalg.lstsq( Cx , NP.eye(Ndata) , rcond = None)[0]
    P = G.T.dot(inv_Cx.dot(G))

    # replace Ch0 if not given
    if X0 is None:
        X0 = NP.zeros(Nh)

    # do a sanity check that all target sigmas have the proper number of elements
    if EPIC_bool is None:
        test = [abs(len(ts) - Npar) for ts in target_sigmas]
    else:
        test = [abs(len(ts) - int(EPIC_bool.sum())) for ts in target_sigmas]

    if NP.sum(test) > 0:
        raise ValueError('some elements of target_sigmas do not have length = Npar')

    # do the calculation for all target sigmas
    for i in range(0, NumTargetSigmas):
        ts = target_sigmas[i]
        print('   ')
        print('*************************************************************')
        print('Step {:d} of {:d}'.format(i + 1, len(target_sigmas)))
        print('** Working on target_sigmas (ts) with : **')
        print('--> ts_min = {:.3f}, ts_max = {:.3f}'.format(NP.min(ts), NP.max(ts)))
        ts = ts.reshape(len(ts))
        epic_sol = calc_EPIC_Ch(P, H, ts, X0, V = V,  LSQpar=LSQpar,
                                homogeneous_step=homogeneous_step,
                                beta_shift_k=beta_shift_k,
                                beta_distance=beta_distance,
                                EPIC_bool = EPIC_bool)
        ChSol.append(epic_sol)

    data_EPIC = {}
    data_EPIC['ChSol'] = ChSol
    data_EPIC['target_sigmas'] = target_sigmas
    
    return data_EPIC


