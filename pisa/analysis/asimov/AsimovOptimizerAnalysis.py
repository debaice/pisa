#! /usr/bin/env python
#
# author: Tim Arlen - tca3@psu.edu
#         Sebatian Boeser - sboeser@uni-mainz.de
#
# date:   02-July-2014

"""
Run the Asimov optimizer-based analysis. Based on LLROptimizerAnalysis, but the
primary difference is that it only uses the one fiducial model template of the
"pseudo data set" and fits to the templates finding the best fit template by
maximizing the LLH / or minimizing the chisquare using the optimizer.
"""

import os
import itertools
from copy import deepcopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import scipy


parser = ArgumentParser(
    description='''Run the Asimov optimizer-based analysis varying the
    systematic parameters; saves the likelihood (or chisquare) values for all
    combinations of hierarchies.''',
    formatter_class=ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    '--template-settings',
    metavar='JSONFILE',
    required=True,
    help='Settings for generating templates.'
)
parser.add_argument(
    '--asimov-data-settings',
    metavar='JSONFILE',
    default=None,
    help='''Settings for generating Asimov data; optional, only need to specify
    if different from template settings.'''
)

parser.add_argument(
    '--sweep-t23',
    metavar='HRLIST_IN_DEG',
    default=None,
    help='''theta23 values (in degrees) to sweep over as injected values for
    Asimov data; forms one dimension in parameter hypercube if other parameters
    are swept'''
)
parser.add_argument(
    '--sweep-dm31',
    metavar='HRLIST_IN_eV2',
    default=None,
    help='''deltam31 values (in eV^2) to sweep over as injected values for
    Asimov data; forms one dimension in parameter hypercube if other parameters
    are swept'''
)
parser.add_argument(
    '--sweep-livetime',
    metavar='HRLIST_IN_YEARS',
    default=None,
    help='''livetime values (in years) to sweep over as injected values for
    Asimov data; forms one dimension in parameter hypercube if other parameters
    are swept'''
)
parser.add_argument(
    '--minimizer-settings',
    metavar='JSONFILE',
    required=True,
    help='Settings related to the optimizer used in the LLR analysis.'
)

parser.add_argument(
    '--metric', choices=['chisquare', 'llh'],
    required=True,
    help='Name of metric to use.'
)
parser.add_argument(
    '--single-octant', action='store_true',
    help='Do NOT check theta23 in other octant after first optimisation.'
)

parser.add_argument(
    '--save-steps', action='store_true',
    help='Save all steps the optimizer takes.'
)
parser.add_argument(
    '--outfile',
    metavar='JSONFILE',
    help='''Output filename. Suffix(es) are added prior to extension if
    sweeping any parameters.'''
)
parser.add_argument(
    '-v', '--verbose', action='count', default=None,
    help='Set verbosity level.'
)
args = parser.parse_args()

from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json, to_json
from pisa.analysis.llr.LLHAnalysis import find_opt_scipy
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy
from pisa.utils import utils

from smartFormat import fnameNumFmt
import genericUtils as GUTIL

set_verbosity(args.verbose)

if args.sweep_livetime is None:
    args.sweep_livetime = [None]
else:
    args.sweep_livetime = GUTIL.hrlist2list(args.sweep_livetime)
if args.sweep_t23 is None:
    args.sweep_t23 = [None]
else:
    args.sweep_t23 = GUTIL.hrlist2list(args.sweep_t23)
if args.sweep_dm31 is None:
    args.sweep_dm31 = [None]
else:
    args.sweep_dm31 = GUTIL.hrlist2list(args.sweep_dm31)

# Read in the settings
template_settings = from_json(args.template_settings)
minimizer_settings = from_json(args.minimizer_settings)
if args.asimov_data_settings is not None:
    asimov_data_settings = from_json(args.asimov_data_settings)
else:
    asimov_data_settings = template_settings

# Workaround for old scipy versions
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0' %scipy.__version__)
    if 'maxiter' in minimizer_settings:
        logging.warn('Optimizer settings for "maxiter" will be ignored')
        minimizer_settings.pop('maxiter')

# Make sure Asimov and template are using the same channel.
channel = template_settings['params']['channel']['value']
if channel != asimov_data_settings['params']['channel']['value']:
    error_msg = 'Both template and pseudo data must have same channel!\n'
    error_msg += ' asimov_data_settings chan: "%s", template chan: "%s" ' \
            %(asimov_data_settings['params']['channel']['value'], channel)
    raise ValueError(error_msg)

template_maker = TemplateMaker(
    get_values(template_settings['params']),
    **template_settings['binning']
)
asimov_data_maker = TemplateMaker(
    get_values(asimov_data_settings['params']),
    **asimov_data_settings['binning']
)

# Now that a bit has been instantiated successfully, create the output dir if
# it doesn't exist
utils.mkdir(os.path.dirname(args.outfile))

for livetime, t23, dm31 in itertools.product(args.sweep_livetime,
                                             args.sweep_t23,
                                             args.sweep_dm31):
    # Set injected sweep parameter(s), if supplied
    labels = []
    if livetime is not None:
        template_settings['params']['livetime']['value'] = livetime
        template_settings['params']['livetime']['fixed'] = True
        labels += ['lt_' + fnameNumFmt(livetime, sigFigs=10,
                                       keepAllSigFigs=False)]
    if t23 is not None:
        template_settings['params']['theta23_nh']['value'] = np.deg2rad(t23)
        template_settings['params']['theta23_nh']['fixed'] = True

        template_settings['params']['theta23_ih']['value'] = np.deg2rad(t23)
        template_settings['params']['theta23_ih']['fixed'] = True
        labels += ['t23_' + fnameNumFmt(t23, sigFigs=10, keepAllSigFigs=False)]
    if dm31 is not None:
        template_settings['params']['deltam31_nh']['value'] = dm31
        template_settings['params']['deltam31_nh']['fixed'] = True

        template_settings['params']['deltam31_ih']['value'] = dm31
        template_settings['params']['deltam31_ih']['fixed'] = True
        labels += ['dm31_' + fnameNumFmt(dm31, sigFigs=10,
                                         keepAllSigFigs=False)]

    # Generate two Asimov (not-fluctuated) datasets, one for each hierarchy.
    # For each Asimov dataset, find the best matching template in each
    # of the hierarchy hypotheses.
    results = {}
    for data_tag, data_normal in [('data_NMH', True), ('data_IMH', False)]:
        results[data_tag] = {}
        # 1) get Asimov data (not fluctuated; i.e., "expected" data)
        asimov_pvals = get_values(
            select_hierarchy(
                asimov_data_settings['params'],
                normal_hierarchy=data_normal
            )
        )

        asimov_fmap = get_asimov_fmap(
            asimov_data_maker,
            asimov_pvals,
            channel=channel
        )

        # 2) Find max llh (or min chisquare) and best-fit free params by matching
        #    templates to Asimov data.
        for hypo_tag, hypo_normal in [('hypo_NMH', True), ('hypo_IMH', False)]:
            physics.info('Finding best fit for %s under %s assumption'
                         %(data_tag, hypo_tag))
            tprofile.info('start optimizer')
            tprofile.info('Using %s' %args.metric)

            opt_data = find_opt_scipy(asimov_fmap, template_maker,
                                      template_settings['params'],
                                      minimizer_settings, args.save_steps,
                                      normal_hierarchy=hypo_normal,
                                      check_octant=not args.single_octant,
                                      metric_name=args.metric)

            tprofile.info('stop optimizer')

            # Store the optimum data
            results[data_tag][hypo_tag] = opt_data

    # Instantiate output dict with settings
    output = {'template_settings': deepcopy(template_settings),
              'minimizer_settings': deepcopy(minimizer_settings)}
    if not utils.recursiveEquality(asimov_data_settings, template_settings):
        output['asimov_data_settings'] = deepcopy(asimov_data_settings),

    output['results'] = results

    # Write to file
    rootname, ext = os.path.splitext(args.outfile)
    if len(labels) > 0:
        labels.insert(0, 'FIXED')
    outfname = rootname + '__'.join(labels) + ext
    to_json(output, outfname)

    # Report result
    if args.metric == 'llh':
        llr_nmh = -(np.min(results['data_NMH']['hypo_IMH'][args.metric])
                    - np.min(results['data_NMH']['hypo_NMH'][args.metric]))
        llr_imh = -(np.min(results['data_IMH']['hypo_IMH'][args.metric])
                    - np.min(results['data_IMH']['hypo_NMH'][args.metric]))
        logging.info('(hypo NMH is numerator): llr_nmh: %.4f, llr_imh: %.4f'
                     %(llr_nmh, llr_imh))

    elif args.metric == 'chisquare':
        chi2_nmh = np.min(results['data_NMH']['hypo_IMH'][args.metric])
        chi2_imh = np.min(results['data_IMH']['hypo_NMH'][args.metric])
        logging.info('chi2_nmh: %.4f, chi2_imh: %.4f' %(chi2_nmh, chi2_imh))
