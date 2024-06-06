# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2020

2022-12-02: - Adds optional minimum norm of diag(Wh) regularization to the calculation 
            of the EPIC.
            - Changes objective function to use target variances as the standard
            deviation of the EPIC in the nonlinear least squares problem.

2024-05-29: to increase speed of the calculations we convert calc_F and calc_JF functions
into a class that acts as a wrapper for the calculation of EPIC residual and its Jacobian
for nonlinear least squares implementation in scipy.optimize.least_squares. The wraper has
a memory of the calculations that are common to the misfit residuals and their Jacobian. 
I thank the people at the Scipy github for giving the great idea of the wrapper at the 
issue post: https://github.com/scipy/scipy/issues/20826#issuecomment-2136326264 

"""
import numpy as np
from scipy.linalg import inv

class objective_fun_wrapper:
    """
    Class that acts as a wrapper for the calculation of EPIC residual and its Jacobian for
    nonlinear least squares implementation in scipy.optimize.least_squares.    
    """
    def __init__(self, P, H, TargetVar, V = None, EPIC_bool = None, regularize = None):
        """
        :param X: unknowns of the function, a 1D numpy array (the betas).
        :param P: precision matrix of the unregularized problem (2D numpy array)
        :param H: regularization operator (2D numpy array)
        :param TargetVar: 1D numpy array with target a posteriori variances
        :param V: if not None, must be a 2D numpy array that can be used to specify a 
            linear relationship between the different variables being search (X).
        :param EPIC_bool: A boolean numpy 1D array indicating which coefficients of m are 
            subject to the EPIC. If EPIC_bool[i] is True then m[i] is subject to EPIC.
            If var_m is the vector with the diagonal elements of the posterior 
            covariance matrix of model parameters (Cm), then, the EPIC is written as:
                        var_m[EPIC_bool] = target_sigmas**2
            CAUTION must be taked when defining EPIC_bool and target_sigmas as the length
            and order of var_m[EPIC_bool] and target_sigmas**2  must match.
        :param regularize: if None, the EPIC condition is solved through an unregularized 
            nonlinear least squares inversion. If a dictionary, can be an empty 
            dictionary, or a dictionary defining 'sigma_weights', the standard deviation
            of the minimum norm prior constraint on the regularization weight. If the
            dictionary does not have the 'sigma_weights' key, the default value is used
            (default : NP.exp(NP.finfo(float).precision/3)).
        
        """
        self.P = P
        self.H = H
        self.TargetVar = TargetVar
        self.V = V
        self.EPIC_bool = EPIC_bool
        self.regularize = regularize
        self.sigma_weight_default =  np.exp(np.finfo(float).precision/4)
        self.X = None
        self.invA = None
        self.beta = None
        self.count_F = 0
        self.count_JF = 0
        self.count_common = 0

    def calc_common(self, X):
        """
        Perform the common calculations for the residuals and the Jacobian.
        If X == self.X, the calculations are not performed and saved results are used.
        """

        if np.array_equal(X, self.X):
            return self.invA, self.beta
        else:
            if self.V is None:
                beta = X
            else:
                beta = self.V.dot(X)
            # assemble the inverse of Ch
            invCh = np.diag(np.exp(beta))
            # compute the EPIC residual vector
            A = self.P + self.H.T @ (invCh @ self.H)
            invA = inv(A)
            # update stored variables that result from the common calculations
            self.invA = invA
            self.beta = beta
            self.X = X

            self.count_common += 1

            return invA, beta
        
    def calc_F(self, X):
        """
        Calculates function F (residuals) of the EPIC

        """
        # compute common calculations
        invA, beta = self.calc_common(X)

        # Assemble F
        if self.EPIC_bool is None:
            F = np.diag(invA) - self.TargetVar
        else:
            F = np.diag(invA)[self.EPIC_bool] - self.TargetVar
        
        F = F / self.TargetVar

        if self.regularize is not None:
            sigma_weight = self.regularize.get('sigma_weight', self.sigma_weight_default)
            F2 = np.exp(beta/2) / sigma_weight
            F = np.hstack((F, F2))
        
        self.count_F += 1
        
        return F      

    def calc_JF(self, X):
        """
        Calculates the Jacobian of the EPIC residuals
        """  
        # compute common calculations
        invA, beta = self.calc_common(X)

        # assemble JF
        # fill the derivatives with respect to each beta
        B = self.H @ invA
        BB = B * B
        E = np.diag(np.exp(beta))
        JF = np.transpose(-1.0 * E @ BB)

        if self.V is not None:
            JF = JF @ self.V

        if self.EPIC_bool is not None:
            JF = JF[self.EPIC_bool, :]

        JF = np.diag(1/self.TargetVar) @ JF

        # extend JF if using regularization to compute EPIC        
        # Note that here the EPIC will be approximately met.
        if self.regularize is not None:
            sigma_weight = self.regularize.get('sigma_weight', self.sigma_weight_default)
            # add the jacobian of the Wh damping 
            JF2 = 0.5 * np.diag(np.exp(beta/2)) / sigma_weight
            JF = np.vstack((JF, JF2))

        self.count_JF += 1

        return JF






