# -*- coding: utf-8 -*-
#
#  RecoServiceKernelFile.py
# 
# Loads a pre-calculated dict of reconstruction kernels stored in json format.
#
# author: Lukas Schulte
#         schulte@physik.uni-bonn.de
#
# date:   August 20, 2014
#

import sys
import logging

from pisa.reco.RecoServiceBase import RecoServiceBase
from pisa.resources.resources import find_resource
from pisa.utils.jsons import from_json


class RecoServiceKernelFile(RecoServiceBase):
    """
    Loads a pre-calculated reconstruction kernel (that has been saved via 
    reco_service.store_kernels) from disk and uses that for reconstruction.
    """
    def __init__(self, ebins, czbins, kernelfile=None, **kwargs):
        """
        Parameters needed to instantiate a reconstruction service with 
        pre-calculated kernels:
        * ebins: Energy bin edges
        * czbins: cos(zenith) bin edges
        * kernelfile: JSON containing the kernel dict
        """
        self.kernelfile = kernelfile
        RecoServiceBase.__init__(self, ebins, czbins, **kwargs)


    def _get_reco_kernels(self, kernelfile=None, **kwargs):
        
        for reco_scale in ['e_reco_scale', 'cz_reco_scale']:
            if reco_scale in kwargs:
                if not kwargs[reco_scale]==1:
                    raise ValueError('%s = %.2f not valid for RecoServiceKernelFile!'
                                     %(reco_scale, kwargs[reco_scale]))
        
        if not kernelfile in [self.kernelfile, None]:
            logging.info('Reconstruction from non-default kernel file %s!'%kernelfile)
            return from_json(find_resource(kernelfile))
        
        if not self.kernels:
            logging.info('Using file %s for default reconstruction'%(kernelfile))
            self.kernels = from_json(find_resource(kernelfile))

        return self.kernels
