#! /usr/bin/env python
#
# BuildFisherMatrix.py
#
# Tools for building the Fisher Matrix from a set of derivatives and a fiducial
# map.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#

import numpy as np

from pisa.analysis.fisher.Fisher import FisherMatrix
from pisa.utils.log import logging

def build_fisher_matrix(gradient_maps, fiducial_map, template_settings, chan):

  # Fix the ordering of parameters
  params = gradient_maps.keys()
  # Find non-empty bins in flattened map
  nonempty = np.nonzero(fiducial_map[chan]['map'].flatten())
  logging.info("Using %u non-empty bins of %u"%(len(nonempty[0]),
                                                  len(fiducial_map[chan]['map'].flatten())))

  # Get gradients as calculated above for non-zero bins
  gradients = np.array([gradient_maps[par]['map'].flatten()[nonempty] for par in params])
  # Get error estimate from best-fit bin count for non-zero bins
  sigmas = np.sqrt(fiducial_map[chan]['map'].flatten()[nonempty])

  # Loop over all parameters per bin (simple transpose) and calculate Fisher
  # matrix per by getting the outer product of all gradients in a bin.
  # Result is sum of matrix for all bins.
  fmatrix = np.zeros((len(params), len(params)))
  for bin_gradients, bin_sigma in zip(gradients.T,sigmas.flatten()):
    fmatrix += np.outer(bin_gradients, bin_gradients)/bin_sigma**2
    
  # And construct the fisher matrix object
  fisher = FisherMatrix(matrix=fmatrix,
                       parameters=params,  #order is important here!
                       best_fits=[template_settings[par]['value'] for par in params],
                       priors=[template_settings[par]['prior'] for par in params],
                       )

  # Return the fisher matrix
  return fisher



