# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Calculation of prior information variances using the EPIC

2022-12-02: - Adds optional minimum norm of diag(Wh) regularization to the calculation 
            of the EPIC.

2024-06-06: - Improves the calculation of the EPIC by using a wrapper class for the misfit
            residuals (calc_F) and their Jacobian (calc_JF). This class has a memory of 
            the calculations that have been done previously and are common to calc_F and 
            calc_JF. This is done to speed up the calculations, making the calculations
            about 30-40% faster.

"""
import numpy as NP
from scipy.linalg import inv
from scipy.optimize import least_squares
from .beta_bounds import compute_bounds
from .objective_fun_wrapper import objective_fun_wrapper

### Main function to calculate Ch using EPIC condition.
def calc_EPIC_Ch(P, H, targetSigma_m, X0 = None, V = None, LSQpar={}, homogeneous_step = False,
                 beta_shift_k = 0, beta_distance = 2, EPIC_bool = None,
                 regularize = None):
    """

    :param P: Precision matrix of the unregularized inverse problem (Nm x Nm)
    :param H: Regularization operator (matrix) (Nh x Nm)
    :param targetSigma_m: target a posteriory standard deviations on model parameters.
                          Must be a single column numpy 2D array. The length must be 
                          equal to the number of model parameters.
    :param X0: initial values of betas, the natural logarithm of the reciprocal of prior 
               information variances (Nh x 1). X0 = -NP.log(Ch0). If not given, will be 
               set to the average of the computed bounds for X = -NP.log(Ch).
    :param LSQpar: a dictionary with several parameters that control convergence of
                nonlinear optimization algorithm used to solve the EPIC condition problem.
    :param V: matrix accounting for a linear variable change, x = V.dot(y) in which
             we search values for y instead of x. Thus X0 must have the dimension of y.
    :param beta_shift_k & beta_distance: see docstring of beta_bounds.compute_bounds
    :param homogeneous_step: if True does first an homogeneous step to find a preliminary
                            initial guess of Ch.
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
                       (default : NP.exp(NP.finfo(float).precision/3)). Note that when
                       regularization is used, the EPIC will be met approximately.
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

    :return: a dictionary with the estimated vector of the natural logarithm of the
             reciprocal of a priori variances Ch and status as well as information on 
             the results of the nonlinear optimization. (See scipy.optimize.least_squares
             output definition for further information).
    """
    Nh, Nm = H.shape

    # set default optimization parameters if not provided
    if Nh > Nm: # There are more prior information variances than number of target sigmas
                # Need to damp iterations.
        msg = '\n**********************************************************\n'
        msg+= '* Warning: Nh > Nm !!!. Iteration steps are being damped *\n'
        msg+= '* and convergence may be slow. Consider using V.         *\n'
        msg+= '**********************************************************\n'
        print(msg)
        # get the parameters needed for least_squares non linear estimation.
        if 'TolX1' not in LSQpar.keys():
            LSQpar['TolX1'] = 1E-6
        if 'TolFun1' not in LSQpar.keys():
            LSQpar['TolFun1'] = 1E-6
        if 'TolG1' not in LSQpar.keys():
            LSQpar['TolG1'] = 1E-6
        if 'TolX2' not in LSQpar.keys():
            LSQpar['TolX2'] = 1E-6
        if 'TolFun2' not in LSQpar.keys():
            LSQpar['TolFun2'] = 1E-6
        if 'TolG2' not in LSQpar.keys():
            LSQpar['TolG2'] = 1E-8
        if 'damp_trf' not in LSQpar.keys():
            LSQpar['damp_trf'] = 1E-3

    else: # case Nh <= Nm

        # get the parameters needed for least_squares non linear estimation.
        if 'TolX1' not in LSQpar.keys():
            LSQpar['TolX1'] = 1E-6
        if 'TolFun1' not in LSQpar.keys():
            LSQpar['TolFun1'] = 1E-6
        if 'TolG1' not in LSQpar.keys():
            LSQpar['TolG1'] = 1E-6
        if 'TolX2' not in LSQpar.keys():
            LSQpar['TolX2'] = 1E-8
        if 'TolFun2' not in LSQpar.keys():
            LSQpar['TolFun2'] = 1E-8
        if 'TolG2' not in LSQpar.keys():
            LSQpar['TolG2'] = 1E-10
        if 'damp_trf' not in LSQpar.keys():
            LSQpar['damp_trf'] = 1E-9

    if 'method' not in LSQpar.keys():
        LSQpar['method'] = 'trf'
    if 'loss' not in LSQpar.keys():
        LSQpar['loss'] = 'linear'
    if 'verbose' not in LSQpar.keys():
        LSQpar['verbose'] = 2

    # set bounds for betas
    bounds = compute_bounds(beta_shift_k, beta_distance)
    # if X0 is not given, average of bounds
    if X0 is None:
        X0 = NP.ones(H.shape[0]) * (bounds[0] + bounds[1]) / 2.0

    # enforce that X0 and targetSigma_m are 1D numpy arrays
    X0 = X0.reshape(len(X0))
    targetSigma_m = targetSigma_m.reshape(len(targetSigma_m))
    
       
    if LSQpar['verbose'] > 0:
            msg = """
            *****************************************************
            ***     Bounds for betas = Log(1/diag(Ch))        ***
            *****************************************************
            *** betas will be within bounds = [{:.2f}, {:.2f}] ***
            *****************************************************
            """
            msg = msg.format(*bounds)
            print(msg)


    # set the a posteriori target variance
    TargetVar = targetSigma_m ** 2
    # set the scipy.optimize.least_squares problem
    # instantiate the F and JF wrapper
    EPICwrapper = objective_fun_wrapper(P = P, H = H, TargetVar = TargetVar, V = V,
                                        EPIC_bool = EPIC_bool,
                                        regularize = regularize)

    # calcF with constant function first to speed up things.
    def calc_F_constantBeta1(x, X0):
        Xtest = x + X0
        return EPICwrapper.calc_F(Xtest)

    # argsFX0 is arguments for  calcF_constantBeta1 
    argsFX0 = (X0,)

    # solve the problem with constant Ch
    x0_4cB = NP.array([0])

    if homogeneous_step:
        if LSQpar['verbose'] > 0:
            msg = """
            ********************************
            *** Searching homogeneous Ch ***
            ********************************
            """
            print(msg)
        sol0 = least_squares(calc_F_constantBeta1, x0_4cB, jac='2-point',
                        method=LSQpar['method'], args=argsFX0,
                        verbose = LSQpar['verbose'], ftol=LSQpar['TolFun1'], 
                        xtol=LSQpar['TolX1'], loss = LSQpar['loss'],
                             gtol = LSQpar['TolG1'],
                             bounds = bounds)
        if LSQpar['verbose'] > 0: print('beta_homogeneous = ', sol0['x'][0])
        Xnext = sol0['x'][0] + X0

        if LSQpar['verbose'] > 0:
            msg = """
            ***************************************************************************
            *** Searching Heterogeneous Ch using Homogeneous Ch as Initial Solution ***
            ***************************************************************************
            """
            print(msg)


    else: # if homogeneous step is not used.
        
        Xnext =  X0
        if LSQpar['verbose'] > 0:
            msg = """
            ***************************************************************************
            *** Searching Heterogeneous Ch                                          ***
            ***************************************************************************
            """
            print(msg)

    Nh, Nm = H.shape
    # just for the counters (delete this line later to avoid one extra computation)
    EPICwrapper = objective_fun_wrapper(P = P, H = H, TargetVar = TargetVar, V = V,
                                        EPIC_bool = EPIC_bool,
                                        regularize = regularize)

    if Nh > Nm: # solve using damped iterations (SLOW!)
        sol = least_squares(EPICwrapper.calc_F, Xnext, 
                            jac=EPICwrapper.calc_JF,
                            method=LSQpar['method'], args=tuple(),
                            verbose=LSQpar['verbose'], ftol=LSQpar['TolFun2'],
                            xtol=LSQpar['TolX2'], loss=LSQpar['loss'],
                            gtol=LSQpar['TolG2'],
                            bounds=bounds,
                            x_scale='jac', tr_solver='lsmr',
                            tr_options={'regularize': True, 'damp': 1e-3})


    else: # case Nh <= Nm
        sol = least_squares(EPICwrapper.calc_F, Xnext, 
                            jac = EPICwrapper.calc_JF,
                            method = LSQpar['method'], args = tuple(), 
                            verbose = LSQpar['verbose'], ftol = LSQpar['TolFun2'], 
                            xtol= LSQpar['TolX2'], loss = LSQpar['loss'],
                            gtol=LSQpar['TolG2'],
                            bounds=bounds, x_scale = 'jac',
                            tr_solver='lsmr',
                            tr_options={'regularize': False, 'damp': LSQpar['damp_trf']})

    if LSQpar['verbose'] > 0:
        print('****************************************************************')
        print('*** calculated betas min max are : ({:.2f}, {:.2f})'.format(
              NP.min(sol['x']), NP.max(sol['x'])))
        #print('Counting common, F and JF')
        #print(EPICwrapper.count_common, EPICwrapper.count_F, EPICwrapper.count_JF)
        print('****************************************************************')

    # clean unnecesary info in sol that uses a lot of memmory
    sol.pop('jac')
    sol.pop('grad')
    sol.pop('active_mask')
    
    
    return sol

