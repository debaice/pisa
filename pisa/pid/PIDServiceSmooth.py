# author: J.L. Lanfranchi
#         jll1062+pisa@phys.psu.edu
#
# date:   March 2, 2016
#
"""
Generate a 2D PID map using samples from smoothed 1D-marginalized PID's (for
now, only smooth is assumed to be in energy).
"""

import numpy as np
from scipy.interpolate import interp1d

from pisa.utils.log import logging, set_verbosity
from pisa.utils import fileio
from pisa.pid.PIDServiceBase import PIDServiceBase


class PIDServiceSmooth(PIDServiceBase):
    """Creates PID maps by interpolation of stored sample points.

    Systematic 'PID_offset' is supported but 'PID_scale' has been
    removed until its implementation is corrected.

    Parameters
    ----------
    ebins, czbins
    pid_energy_smooth
    PID_offset
    """
    def __init__(self, ebins, czbins, pid_energy_smooth, PID_offset=0,
                 **kwargs):
        super(PIDServiceSmooth, self).__init__(ebins, czbins)
        self.pid_energy_smooth = None
        self.esmooth_pid_data = None
        self.esmooth_ebin_midpoints = None
        self.esmooth_interpolants = None
        self.PID_offset = None
        self.pid_kernels = None
        self.update(ebins=ebins, czbins=czbins,
                    pid_energy_smooth=pid_energy_smooth,
                    PID_offset=PID_offset)

    def update(self, ebins=None, czbins=None, pid_energy_smooth=None,
               PID_offset=None, interp_kind='cubic'):
        if pid_energy_smooth is None:
            pid_energy_smooth = self.pid_energy_smooth
        if ebins is None:
            ebins = self.ebins
        if czbins is None:
            czbins = self.czbins

        # Don't do anything if not necessary
        if self.pid_kernels is not None and np.all(ebins == self.ebins) and \
                np.all(czbins == self.czbins) and \
                np.all(pid_energy_smooth == self.pid_energy_smooth) and \
                PID_offset == self.PID_offset:
            return self.pid_kernels

        self.ebins = ebins
        self.czbins = czbins
        self.PID_offset = PID_offset
        ebin_midpoints = (ebins[:-1] + ebins[1:]) / 2.0
        n_czbins = len(self.czbins) - 1

        # Load smoothing in energy
        new_file = False
        if self.esmooth_pid_data is None or \
                pid_energy_smooth != self.pid_energy_smooth:
            new_file = True
            logging.info('Loading smoothed-PID-energy-dependence file'
                         ' %s' % pid_energy_smooth)
            self.esmooth_pid_data = fileio.from_file(pid_energy_smooth)
            self.esmooth_ebin_midpoints = \
                    self.esmooth_pid_data.pop('ebin_midpoints')
            self.labels = sorted(self.esmooth_pid_data.keys())
            self.signatures = self.esmooth_pid_data[self.labels[0]].keys()

        # (Re)create interpolants if necessary
        if self.esmooth_interpolants is None or new_file:
            self.esmooth_interpolants = {}
            for label in self.labels:
                pid_allocations = self.esmooth_pid_data[label]
                self.esmooth_interpolants[label] = {}
                for sig, allocation in pid_allocations.iteritems():
                    self.esmooth_interpolants[label][sig] = interp1d(
                        x=self.esmooth_ebin_midpoints,
                        y=self.esmooth_pid_data[label][sig]['smooth'],
                        kind=interp_kind,
                        copy=False,
                        bounds_error=True,
                        fill_value=np.nan,
                        #assume_sorted=True,
                    )

        # Assume no variation in PID across coszen
        coszen_smooth = np.ones(n_czbins)

        self.pid_kernels = {
            'binning': {'ebins': self.ebins, 'czbins': self.czbins}
        }
        for label in self.labels:
            self.pid_kernels[label] = {}
            for sig in self.signatures:
                energy_smooth = np.clip(
                    self.esmooth_interpolants[label][sig](ebin_midpoints)
                    - PID_offset, a_min=0, a_max=1
                )
                kernel = np.outer(energy_smooth, coszen_smooth)
                self.pid_kernels[label][sig] = kernel

        return self.pid_kernels

    def __get_pid_kernels(self, pid_energy_smooth=None, PID_offset=None):
        return self.update(ebins=self.ebins, czbins=self.czbins,
                           pid_energy_smooth=pid_energy_smooth,
                           PID_offset=PID_offset)

    @staticmethod
    def add_argparser_args(parser):
        parser.add_argument(
            '--pid-energy-smooth', metavar='RESOURCE_NAME', type=str,
            default='pid/pingu_v36/'
            'pid_energy_smooth__pingu_v36__runs_388-390__proc_5__pid_1.json',
            help='''[ PID-Smooth ] JSON file containing smoothed PID
            energy smooth for each particle signature'''
        )
        return parser
