# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
frortega@gmail.com
Departamento de Geofísica - FCFM
Universidad de Chile

2021

Module to save and read precomputed values of EPIC Ch using python pickle.
TODO: Handle verbosity.
"""
import pickle

def save_precomputed_Ch(filename, data_EPIC, pickle_protocol = None):
    """
    Saves the dictionary data_EPIC created with precompute_EPIC_Ch info a file.
    As the dictionary data_EPIC can use a large amount of memory, and original object
    and pickled object must reside at some poing in memory, in order to be more
    efficient with the use of resources  I take EPIC results and save each results as
    consecutive dumps in the pickle file.

    dump 1: an integer NTS: number of target sigmas (also # of epic solutions)
    dump 2: a list with NTS elements which are the target_sigmas vectors.
    dump 3 to NTS+3: each solution of EPIC_Ch in the same order as in list of dump 2

    The code assumes the following structure for data_EPIC
    data_EPIC = {}
    data_EPIC['ChSol'] = ChSol # ChSol is a list with solutions for each target_sigma
    data_EPIC['target_sigmas'] = target_sigmas

    :param pickle_protocol: if None, uses pickle.HIGHEST_PROTOCOL, otherwise, integer
    must specify desired valid protocol. This is provided with compatibility to save
    information for previous python versions.

    """
    if pickle_protocol is None:
        pickle_protocol = pickle.HIGHEST_PROTOCOL
    Ch_Sol = data_EPIC['ChSol']
    target_sigmas = data_EPIC['target_sigmas']
    NTS = len(Ch_Sol)
    print('\nSaving {:d} EPIC Ch solutions to file {:s}\n'.format(NTS, filename))

    with open(filename, 'wb') as picklefile:
        # dump 1 : NTS
        pickle.dump(NTS, picklefile, protocol=pickle_protocol)
        # dump 2 : target_sigmas
        pickle.dump(target_sigmas, picklefile, protocol=pickle_protocol)
        # dump 3 : Ch_EPIC solutions.
        for Ch in Ch_Sol:
            pickle.dump(Ch, picklefile, protocol=pickle_protocol)

    return None


def read_precomputed_Ch(filename):
    """
    Reads precomputed values of EPIC Ch saved using save_precomputed_Ch into the pickle
    file "filename".

    :param filename: name of the pickle file generated with save_precomputed_Ch
    :return: the dictionary data_EPIC
    data_EPIC = {}
    data_EPIC['ChSol'] = ChSol # ChSol is a list with solutions for each target_sigma
    data_EPIC['target_sigmas'] = target_sigmas
    """
    with open(filename, 'rb') as picklefile:
        NTS = pickle.load(picklefile)
        target_sigmas = pickle.load(picklefile)
        Ch_Sol = [pickle.load(picklefile) for k in range(0, NTS)]

    data_EPIC = {}
    data_EPIC['ChSol'] = Ch_Sol
    data_EPIC['target_sigmas'] = target_sigmas

    return data_EPIC
