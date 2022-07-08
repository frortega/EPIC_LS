# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2021

Module to assemble extended d vector and G, Cx matrices for the inversion problem where
both EPIC (H, ho) and nonEPIC (H_ne, ho_ne, Ch_ne) regularizations are used to define
prior information (see precompute_EPIC_Ch.py for an explanation of such variables.)

"""
import numpy as NP
from scipy.linalg import block_diag


def assemble_extended_d_G_Cx(G, Cx, H_ne, Ch_ne, d = None, ho_ne = None):
    """
    Assembles the extended versions of G, d, and Cx for the inversion problem where 
    both EPIC and nonEPIC regularizations are used to define prior information. 
    - G, H_ne and Ch_ne matrices must be 2D numpy arrays
    - d and ho_ne vectors must be 1 column 2D numpy arrays.
    
    Returns:
    
        G_extended, Cx_extended                     if ho_ne or d are None
        G_extended, Cx_extended, d_extended         otherwise

        
    """
    Ndata, Npar = G.shape
    Nh_ne, Npar3 = H_ne.shape
    # check that G and H apply to the same number of parameters
    if Npar != Npar3:
        raise ValueError('G, and H_ne must have the same number of columns!...')

    # as H_ne.dot(m) = ho_ne is not subject to the EPIC, is appended below G.
    G_extended = NP.vstack((G, H_ne))
    # also, Cx needs to be extended using block_diag
    Cx_extended = block_diag(Cx, Ch_ne)

    if ho_ne is None or d is None:
        return G_extended, Cx_extended

    else:
        # consistently ho_ne is appended below d
        d_extended = NP.vstack((d, ho_ne))
        return G_extended, Cx_extended, d_extended