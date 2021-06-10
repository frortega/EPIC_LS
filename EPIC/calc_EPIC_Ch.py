# -*- coding: utf-8 -*-
__doc__ = """
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

Calculation of prior information variances using the EPIC

"""
import numpy as NP
from scipy.linalg import inv
from scipy.optimize import least_squares
from .beta_bounds import compute_bounds
from .F_JF import calc_F, calc_JF
from numpy.linalg import lstsq

### Main function to calculate Ch using EPIC condition.
def calc_EPIC_Ch(P, H, targetSigma_m, X0, V = None, LSQpar={}, homogeneous_step = False,
                 shift_k = 0):
    """

    :param P: Precision matrix of the unregularized inverse problem (Nm x Nm)
    :param H: Regularization operator (matrix) (Nh x Nm)
    :param targetSigma_m: target a posteriory error on model parameters (sqrt(variance)).
    :param X0: initial values of betas, the natural logarithm of the reciprocal of prior 
               information variances (Nh x 1). X0 = -NP.log(Ch0). 
    :param LSQpar: a dictionary with several parameters that control convergence of
                nonlinear optimization algorithm used to solve the EPIC condition problem.
    :param V: matrix accounting for a linear variable change, x = V.dot(y) in which
             we search values for y instead of x. Thus X0 must have the dimension of y.
    :param shift_k: see docstring of beta_bounds.compute_bounds
    :param homogeneous_step: if True does homogeneous step to find an Homogeneous Ch.

    LSQpar must be a dictionary containing the convergence parameters:    
        - LSQpar['TolX1'] = 1e5 * eps :
        - LSQpar['TolFun1'] = 1e5 * eps
        - LSQpar['TolX2'] = 1e3 * eps
        - LSQpar['TolFun2'] = 1e3 * eps
        - LSQpar['method'] = 'trf'
        - LSQpar['verbose'] = 2
        see scipy.optimize.least_squares help for further information.

    :return: a dictionary with the estimated vector of the natural logarithm of the
             reciprocal of a priori variances Ch and status as well as information on 
             the results of the nonlinear optimization.
    """
    # enforce that X0 and targetSigma_m are 1D numpy arrays
    X0 = X0.reshape(len(X0))
    targetSigma_m = targetSigma_m.reshape(len(targetSigma_m))
    # set bounds for betas
    bounds = compute_bounds(shift_k)
    print(' betas will be within bounds = [{:.2f}, {:.2f}]'.format(
       *bounds))

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

    else: # case Nh <= Nm

        # get the parameters needed for least_squares non linear estimation.
        if 'TolX1' not in LSQpar.keys():
            LSQpar['TolX1'] = 1E-6
        if 'TolFun1' not in LSQpar.keys():
            LSQpar['TolFun1'] = 1E-6
        if 'TolG1' not in LSQpar.keys():
            LSQpar['TolG1'] = 1E-6
        if 'TolX2' not in LSQpar.keys():
            LSQpar['TolX2'] = 1E-4
        if 'TolFun2' not in LSQpar.keys():
            LSQpar['TolFun2'] = 1E-4
        if 'TolG2' not in LSQpar.keys():
            LSQpar['TolG2'] = 1E-6

    if 'method' not in LSQpar.keys():
        LSQpar['method'] = 'trf'
    if 'loss' not in LSQpar.keys():
        LSQpar['loss'] = 'linear'
    if 'verbose' not in LSQpar.keys():
        LSQpar['verbose'] = 2


    # set the a posteriori target variance
    TargetVar = targetSigma_m ** 2
    # set the scipy.optimize.least_squares problem
    Fargs = (P, H, TargetVar, V)

    # calcF with constant function first to speed up things.
    def calc_F_constantBeta1(x, X0, P, H, TargetVar, V = None):
        Xtest = x + X0
        return calc_F(Xtest, P, H, TargetVar, V)

    # argsFX0 is arguments for  calcF_constantBeta1
    argsFX0 = (X0, P, H, TargetVar, V)

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
        print('beta0 = ', sol0['x'][0]/2)
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

    if Nh > Nm: # solve using damped iterations (SLOW!)
        sol = least_squares(calc_F, Xnext, jac=calc_JF,
                            method=LSQpar['method'], args=Fargs,
                            verbose=LSQpar['verbose'], ftol=LSQpar['TolFun2'],
                            xtol=LSQpar['TolX2'], loss=LSQpar['loss'],
                            gtol=LSQpar['TolG2'],
                            bounds=bounds,
                            x_scale='jac', tr_solver='lsmr',
                            tr_options={'regularize': True, 'damp': 1e-3})


    else: # case Nh <= Nm
        sol = least_squares(calc_F, Xnext, jac = calc_JF,
                        method = LSQpar['method'], args = Fargs, 
                        verbose = LSQpar['verbose'], ftol = LSQpar['TolFun2'], 
                        xtol= LSQpar['TolX2'], loss = LSQpar['loss'],
                        gtol=LSQpar['TolG2'],
                        bounds=bounds, x_scale = 'jac')

    print(' betas bounds = [{:.2f}, {:.2f}]'.format(
       *bounds))
    print('calculated betas min max are : ({:.2f}, {:.2f})'.format(
        NP.min(sol['x']), NP.max(sol['x'])))

    return sol

