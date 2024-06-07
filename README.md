# EPIC Tikhonov Regularization for Least Squares Inversion (EPIC_LS)

This package allows the user to perform a (Linear) General Least Squares Inversion with EPIC Tikhonov regularization. 
EPIC stands for Equal Posterior Information Condition (Ortega-Culaciati et al., 2021).

EPIC_LS includes 2 subpackages:

- EPIC: codes to compute the EPIC for the general linear least squares problem.
- LeastSquares: codes to solve the linear least squares problem and the general linear least squares problem (with linear regularization). As an option, non negativity constraints can be applied to all model parameters.

#### PLEASE CITE:
- The article describing the details of the methodology and notation of the codes: 
Ortega-Culaciati, F., Simons, M., Ruiz, J., Rivera, L., & Diaz-Salazar, N. (2021). An EPIC Tikhonov regularization: Application to quasi-static fault slip inversion. Journal of Geophysical Research: Solid Earth, 126, e2020JB021141. https://doi.org/10.1029/2020JB021141 (Also see CITATION.bib).

#### EXAMPLES:
- Examples on how to use the EPIC_LS package will be added soon to this repository.

-----
#### IMPORTANT TIPS: 
- The calculation time will decrease considerably if scipy is compiled against mkl openblas (or similar) cappable of using multicore (default in anaconda python scipy). Remember to check environment variables OMP_NUM_THREADS, MKL_NUM_THREADS, MKL_DOMAIN_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS to have the right number of cores of your machine.

- To use this package add this folder to your PYTHONPATH environment variable.

-----
#### LATEST UPDATES:
- June 6, 2024: Changes calc_EPIC_Ch.py for improved efficiency (Ch estimation now runs about 30% faster).

- December 7, 2022: Adds the option to define a regularization term that is not subject to the EPIC (see variable H_ne, Ch_ne in precompute_EPIC_Ch.py). Additionally, adds an option "regularize" in precompute_EPIC_Ch.py to allow a minimum norm regularization of the EPIC weights (the reciprocal of
the standard deviations computed for the prior information using the EPIC). See "regularize" option in precompute_EPIC_Ch.py.

- July 7, 2022: Adds the option to define only a subset of parameters that are subject to the EPIC (see variable EPIC_bool in calc_EPIC_Ch.py)

- October 26, 2021: Adds the option for performing the Least Squares Estimation applying non negativity constraints on all model parameters (see LeastSquaresNonNeg.py and LeastSquaresRegNonNeg.py in EPIC_LS/LeastSquares/).
