# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2026

Modules to precompute EPIC Ch for a Least Squares Problem, and store it into a file.

Parallel (multiprocessing.Pool) variant of precompute_EPIC_Ch.py; 

Adds an num_proc parameter to precompute_EPIC_Ch, _precompute_EPIC_Ch and
_precompute_EPIC_Ch_HnoEPIC: each target_sigma is solved as an independent
task in a multiprocessing.Pool (chunksize=1), with shared read-only arguments bound 
via functools.partial (no global state, no Pool initializer). num_proc<=1 (or None resolving
 to 1) runs the exact original sequential loop. The psutil and threadpoolctl are
 hard dependencies: psutil determines the default num_proc (physical CPU core count);
 threadpoolctl.threadpool_limits(limits=1) scopes internal numpy/scipy threads to 1 around
each calc_EPIC_Ch call inside a worker, to avoid oversubscription.

"""
import io
import contextlib
import functools
import multiprocessing
import os
import numpy as NP
import psutil
import threadpoolctl
from tqdm import tqdm
import scipy as SP
from scipy.linalg import block_diag
from .calc_EPIC_Ch import calc_EPIC_Ch
from .partial_EPIC_problem import assemble_extended_d_G_Cx


def _default_num_proc():
    """Physical CPU cores, falling back to logical cores, then 1."""
    return psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1


class _TeeWriter:
    """Writes to an in-memory buffer and, if given, flushes each write to a log file too."""
    def __init__(self, buf, log_file=None):
        self._buf = buf
        self._log_file = log_file

    def write(self, s):
        self._buf.write(s)
        if self._log_file is not None:
            self._log_file.write(s)
            self._log_file.flush()  # so `tail -f` sees output as soon as it is printed
        return len(s)

    def flush(self):
        if self._log_file is not None:
            self._log_file.flush()


def _print_summary_table(ChSol, target_sigmas):
    """Prints one row per target_sigma with beta/bound stats and the solved objective cost."""
    header = '{:>6s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>14s} {:>9s} {:>13s}'.format(
        'index', 'ts_min', 'ts_max', 'bound_lo', 'beta_min', 'beta_max', 'bound_hi', 'final_cost',
        'attempts', 'beta_shift_k')
    print(header)
    print('-' * len(header))
    for i, (sol, ts) in enumerate(zip(ChSol, target_sigmas)):
        ts = NP.asarray(ts).reshape(-1)
        print('{:6d} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:12.3f} {:14.6e} {:9d} {:13.3f}'.format(
            i, NP.min(ts), NP.max(ts), sol['beta_bounds'][0], sol['beta_min'], 
            sol['beta_max'], sol['beta_bounds'][1], sol['cost'], sol['beta_shift_attempts'],
            sol['beta_shift_k']))


def _epic_task(item, P, H, X0, V, LSQpar, homogeneous_step, beta_shift_k,
               beta_distance, EPIC_bool, regularize, beta_margin, max_retries,
               log_dir=None, index_width=1):
    """Runs one calc_EPIC_Ch call; captures its stdout for ordered replay by the parent.

    If log_dir is not None, also live-writes (flushed per print) to a per-index log file
    so progress can be watched with `tail -f` while the computation is still running.
    """
    index, ts = item
    buf = io.StringIO()
    log_path = None
    if log_dir is not None:
        log_path = os.path.join(log_dir, 'step_{:0{w}d}.log'.format(index, w=index_width))
    with contextlib.ExitStack() as stack:
        log_file = stack.enter_context(open(log_path, 'w')) if log_path is not None else None
        stack.enter_context(contextlib.redirect_stdout(_TeeWriter(buf, log_file)))
        print('   ')
        print('*************************************************************')
        print('Step {:d} (index {:d})'.format(index + 1, index))
        print('** Working on target_sigmas (ts) with : **')
        print('--> ts_min = {:.3f}, ts_max = {:.3f}'.format(NP.min(ts), NP.max(ts)))
        ts = ts.reshape(len(ts))
        # limit BLAS/OpenMP threads only around the solve, not the prints above
        with threadpoolctl.threadpool_limits(limits=1):
            epic_sol = calc_EPIC_Ch(P, H, ts, X0, V=V, LSQpar=LSQpar,
                                     homogeneous_step=homogeneous_step,
                                     beta_shift_k=beta_shift_k,
                                     beta_distance=beta_distance,
                                     EPIC_bool=EPIC_bool,
                                     regularize=regularize,
                                     beta_margin=beta_margin,
                                     max_retries=max_retries)
    return index, epic_sol, buf.getvalue()


def precompute_EPIC_Ch_pool(G, Cx, H, target_sigmas, X0 = None, V = None,
            LSQpar={}, homogeneous_step = True,
            beta_shift_k=0, beta_distance=2,
            EPIC_bool = None,
            H_ne = None, Ch_ne = None,
            regularize = None,
            num_proc = None,
            log_dir = './log_EPIC_Ch',
            verbosity = 1,
            beta_margin = 0.5, max_retries = 3):
    """

    :param G: Design matrix with Green's functions of the problem (Nd x Nm)
    :param Cx: Misfit covariance matrix (Nd x Nd)
    :param H: Regularization operator (matrix) (Nh x Nm). This H is subject to the EPIC.
    :param H_ne: Regularization operator that is NOT subject to the EPIC.
    :param Ch_ne: Covariance matrix of prior information represented via H_ne.dot(m)=ho_ne
                  where h_ne is the mean of the prior information that is NOT subject
                  to the EPIC. if H_ne is not None, Ch_ne must be a numpy 2D array.
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
    :param num_proc: number of worker processes used to solve different target_sigmas
                   in parallel (one process-pool task per target_sigma, chunksize=1).
                   If None (default), uses the physical CPU core count (via psutil).
                   If <= 1, runs sequentially exactly as the unparallelized code does,
                   with no multiprocessing.Pool overhead. Inside each worker, BLAS/OpenMP
                   threads are limited to 1 via threadpoolctl.threadpool_limits, scoped
                   around the calc_EPIC_Ch call, to avoid oversubscription; results
                   are returned in target_sigmas order regardless of completion order.
    :param log_dir: if not None, path to a folder (created if it does not exist) where one
                   'step_XXXX.log' file per target_sigma is written, flushed after every
                   print so it can be tailed (e.g. `tail -f`) while the computation is
                   still running. Log files always contain full solver iteration detail
                   (LSQpar['verbose'] is forced to 2 for them) regardless of verbosity.
                   If None, no log files are written. Default value is './log_EPIC_Ch'.
    :param verbosity: controls how much is printed while iterating over target_sigmas.
                   0 = nothing is printed; 1 = a tqdm progress bar plus a summary table
                   at the end; 2 = the original step-by-step prints (and any
                   LSQpar['verbose'] driven solver logs) plus the summary table at the
                   end. Must be one of {0, 1, 2}.
    :param beta_margin & max_retries: see docstring of calc_EPIC_Ch.

    :return: a list in which each item is a dictionary with the estimated vector of a
    priori variances Ch and status information on the results of the nonlinear
    optimization used to calculate the EPIC condition. The order of the list is the same
    order in which
    """
    if verbosity not in (0, 1, 2):
        raise ValueError('verbosity must be one of 0, 1, 2 !!!')

    if H_ne is None: # only EPIC regularization is used.
        return _precompute_EPIC_Ch(G, Cx, H, target_sigmas, X0 = X0, V = V,
                                   LSQpar = LSQpar, 
                                   homogeneous_step = homogeneous_step,
                                   beta_shift_k = beta_shift_k,
                                   beta_distance = beta_distance,
                                   EPIC_bool = EPIC_bool,
                                   regularize = regularize,
                                   num_proc = num_proc,
                                   log_dir = log_dir,
                                   verbosity = verbosity,
                                   beta_margin = beta_margin,
                                   max_retries = max_retries)

    else: # EPIC and NON EPIC regularization are used.
        if Ch_ne is None:
            raise ValueError('Ch_ne must be different from None !!!')
        else:
            return _precompute_EPIC_Ch_HnoEPIC(G, Cx, H_ne, Ch_ne, H, target_sigmas, 
                                               X0 = X0, V = V,
                                               LSQpar = LSQpar, 
                                               homogeneous_step = homogeneous_step,
                                               beta_shift_k = beta_shift_k,
                                               beta_distance = beta_distance,
                                               EPIC_bool = EPIC_bool,
                                               regularize = regularize,
                                               num_proc = num_proc,
                                               log_dir = log_dir,
                                               verbosity = verbosity,
                                               beta_margin = beta_margin,
                                               max_retries = max_retries)




#### precompute EPIC also including NON EPIC regularization (H_ne).
def _precompute_EPIC_Ch_HnoEPIC(G, Cx, H_ne, Ch_ne, H, target_sigmas, X0 = None, V = None,
            LSQpar={}, homogeneous_step = True,
            beta_shift_k=0, beta_distance=2,
            EPIC_bool = None,
            regularize = None,
            num_proc = None,
            log_dir = None,
            verbosity = 1,
            beta_margin = 0.5, max_retries = 3):
    """

    :param G: Design matrix with Green's functions of the problem (Nd x Nm)
    :param Cx: Misfit covariance matrix (Nd x Nd)
    :param H_ne: Regularization operator that is NOT subject to the EPIC.
    :param Ch_ne: Covariance matrix of prior information represented via H_ne.dot(m)=ho_ne
                  where h_ne is the mean of the prior information that is NOT subject
                  to the EPIC. if H_ne is not None, Ch_ne must be a numpy 2D array.
    :param H: Regularization operator (matrix) (Nh x Nm) subject to the EPIC.
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
    :param num_proc: number of worker processes used to solve different target_sigmas
                   in parallel (one process-pool task per target_sigma, chunksize=1).
                   If None (default), uses the physical CPU core count (via psutil).
                   If <= 1, runs sequentially exactly as the unparallelized code does,
                   with no multiprocessing.Pool overhead. Inside each worker, BLAS/OpenMP
                   threads are limited to 1 via threadpoolctl.threadpool_limits, scoped
                   around the calc_EPIC_Ch call, to avoid oversubscription; results
                   are returned in target_sigmas order regardless of completion order.
    :param log_dir: see precompute_EPIC_Ch_pool.
    :param verbosity: see precompute_EPIC_Ch_pool.
    :param beta_margin & max_retries: see docstring of calc_EPIC_Ch.

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
    Nh_ne, Npar3 = H_ne.shape
    # check that G and H apply to the same number of parameters
    if Npar != Npar2 or Npar != Npar3 or Npar2 != Npar3:
        raise ValueError('G, H and H_ne must have the same number of columns!...')

    # get the extended equivalent problem
    G_extended, Cx_extended = assemble_extended_d_G_Cx( G = G, 
                                                        Cx = Cx, 
                                                        H_ne = H_ne, 
                                                        Ch_ne = Ch_ne)
    
    # precision matrix of the unregularized problem
    Ndata_extended = Ndata + Nh_ne
    inv_Cx = NP.linalg.lstsq( Cx_extended , NP.eye(Ndata_extended) , rcond = None)[0]
    P = G_extended.T.dot(inv_Cx.dot(G_extended))


    # do a sanity check that all target sigmas have the proper number of elements
    if EPIC_bool is None:
        test = [abs(len(ts) - Npar) for ts in target_sigmas]
    else:
        test = [abs(len(ts) - int(EPIC_bool.sum())) for ts in target_sigmas]

    if NP.sum(test) > 0:
        raise ValueError('some elements of target_sigmas do not have length = Npar')

    # resolve degree of parallelism
    if num_proc is None:
        num_proc = _default_num_proc()
    num_proc = max(1, min(int(num_proc), NumTargetSigmas))

    # local copy so the caller's LSQpar is not mutated; silence solver logs unless verbosity==2
    LSQpar = dict(LSQpar)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        LSQpar['verbose'] = 2  # log files always keep full solver detail regardless of verbosity
    elif verbosity < 2:
        LSQpar['verbose'] = 0
    index_width = len(str(max(NumTargetSigmas - 1, 0)))
    task_fn = functools.partial(_epic_task, P=P, H=H, X0=X0, V=V, LSQpar=LSQpar,
                                 homogeneous_step=homogeneous_step,
                                 beta_shift_k=beta_shift_k,
                                 beta_distance=beta_distance,
                                 EPIC_bool=EPIC_bool,
                                 regularize=regularize,
                                 beta_margin=beta_margin,
                                 max_retries=max_retries,
                                 log_dir=log_dir,
                                 index_width=index_width)

    if num_proc <= 1:
        for i in tqdm(range(0, NumTargetSigmas), disable=(verbosity != 1), desc='EPIC Ch'):
            ts = target_sigmas[i]
            _, epic_sol, captured_text = task_fn((i, ts))
            if verbosity == 2:
                print(captured_text, end='')
            ChSol.append(epic_sol)
    else:
        # one process-pool task per target_sigma; chunksize=1 favors dynamic
        # work-stealing since solve times vary widely between target_sigmas
        ChSol = [None] * NumTargetSigmas
        with multiprocessing.Pool(processes=num_proc) as pool:
            for index, epic_sol, captured_text in tqdm(
                    pool.imap_unordered(task_fn, list(enumerate(target_sigmas)), chunksize=1),
                    total=NumTargetSigmas, disable=(verbosity != 1), desc='EPIC Ch'):
                if verbosity == 2:
                    print(captured_text, end='')
                ChSol[index] = epic_sol

    data_EPIC = {}
    data_EPIC['ChSol'] = ChSol
    data_EPIC['target_sigmas'] = target_sigmas

    if verbosity >= 1:
        _print_summary_table(ChSol, target_sigmas)

    return data_EPIC


#### precompute EPIC with only EPIC Tikhonov regularization.
def _precompute_EPIC_Ch(G, Cx, H, target_sigmas, X0 = None, V = None,
            LSQpar={}, homogeneous_step = True,
            beta_shift_k=0, beta_distance=2,
            EPIC_bool = None,
            regularize = None,
            num_proc = None,
            log_dir = None,
            verbosity = 1,
            beta_margin = 0.5, max_retries = 3):
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
    :param num_proc: number of worker processes used to solve different target_sigmas
                   in parallel (one process-pool task per target_sigma, chunksize=1).
                   If None (default), uses the physical CPU core count (via psutil).
                   If <= 1, runs sequentially exactly as the unparallelized code does,
                   with no multiprocessing.Pool overhead. Inside each worker, BLAS/OpenMP
                   threads are limited to 1 via threadpoolctl.threadpool_limits, scoped
                   around the calc_EPIC_Ch call, to avoid oversubscription; results
                   are returned in target_sigmas order regardless of completion order.
    :param log_dir: see precompute_EPIC_Ch_pool.
    :param verbosity: see precompute_EPIC_Ch_pool.
    :param beta_margin & max_retries: see docstring of calc_EPIC_Ch.

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

    # do a sanity check that all target sigmas have the proper number of elements
    if EPIC_bool is None:
        test = [abs(len(ts) - Npar) for ts in target_sigmas]
    else:
        test = [abs(len(ts) - int(EPIC_bool.sum())) for ts in target_sigmas]

    if NP.sum(test) > 0:
        raise ValueError('some elements of target_sigmas do not have length = Npar')

    # resolve degree of parallelism
    if num_proc is None:
        num_proc = _default_num_proc()
    num_proc = max(1, min(int(num_proc), NumTargetSigmas))

    # local copy so the caller's LSQpar is not mutated; silence solver logs unless verbosity==2
    LSQpar = dict(LSQpar)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        LSQpar['verbose'] = 2  # log files always keep full solver detail regardless of verbosity
    elif verbosity < 2:
        LSQpar['verbose'] = 0
    index_width = len(str(max(NumTargetSigmas - 1, 0)))
    task_fn = functools.partial(_epic_task, P=P, H=H, X0=X0, V=V, LSQpar=LSQpar,
                                 homogeneous_step=homogeneous_step,
                                 beta_shift_k=beta_shift_k,
                                 beta_distance=beta_distance,
                                 EPIC_bool=EPIC_bool,
                                 regularize=regularize,
                                 beta_margin=beta_margin,
                                 max_retries=max_retries,
                                 log_dir=log_dir,
                                 index_width=index_width)

    if num_proc <= 1:
        for i in tqdm(range(0, NumTargetSigmas), disable=(verbosity != 1), desc='EPIC Ch'):
            ts = target_sigmas[i]
            _, epic_sol, captured_text = task_fn((i, ts))
            if verbosity == 2:
                print(captured_text, end='')
            ChSol.append(epic_sol)
    else:
        # one process-pool task per target_sigma; chunksize=1 favors dynamic
        # work-stealing since solve times vary widely between target_sigmas
        ChSol = [None] * NumTargetSigmas
        with multiprocessing.Pool(processes=num_proc) as pool:
            for index, epic_sol, captured_text in tqdm(
                    pool.imap_unordered(task_fn, list(enumerate(target_sigmas)), chunksize=1),
                    total=NumTargetSigmas, disable=(verbosity != 1), desc='EPIC Ch'):
                if verbosity == 2:
                    print(captured_text, end='')
                ChSol[index] = epic_sol

    data_EPIC = {}
    data_EPIC['ChSol'] = ChSol
    data_EPIC['target_sigmas'] = target_sigmas

    if verbosity >= 1:
        _print_summary_table(ChSol, target_sigmas)

    return data_EPIC



