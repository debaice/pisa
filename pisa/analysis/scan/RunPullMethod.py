#! /usr/bin/env python
#
# RunPullMethod.py
#
# Scans over the space of atmospheric (theta23) oscillation parameters,
# analytically minimising the chisquare over the other - linearised - parameters
# by calculating the Fisher matrix at each point on the grid.
#
# author: Thomas Ehrhardt - tehrhard@uni-mainz.de
#
# date:   25-May-2015
#

import sys
import numpy as np
from itertools import product
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.analysis.llr.LLHAnalysis import find_min_chisquare_bfgs
from pisa.analysis.stats.LLHStatistics import get_binwise_chisquare
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.scan.Scan import calc_steps
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.analysis.fisher.FisherAnalysis import get_fisher_matrices

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_atm_params, get_atm_params
from pisa.utils.utils import Timer

parser = ArgumentParser(
    description='''Pull method.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = False,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-g','--grid_settings',type=str,metavar='JSONFILE', required = True,
                    help='''Get chisquare at defined oscillation parameter grid values,
                    according to these input settings.''')
parser.add_argument('-s','--save-steps',action='store_true',default=False,
                    dest='save_steps',
                    help='''Save all steps the optimizer takes.''')
parser.add_argument('--gpu_id',type=int,default=None,
                    help="GPU ID if available.")
#parser.add_argument('-c','--chan',type=str,default='all',
#                    choices=['trck','cscd','all','no_pid'],
#                    help='''which channel to use in the fit.''')
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help='''Output filename.''')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='''set verbosity level''')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
hypo_settings = from_json(args.template_settings)
minimizer_settings = from_json(args.minimizer_settings) if args.minimizer_settings is not None else None
grid_settings = from_json(args.grid_settings)

channel = template_settings['params']['channel']['value']
#Workaround for old scipy versions
import scipy
if scipy.__version__ < '0.12.0':
    logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
    if 'maxiter' in minimizer_settings:
        logging.warn('Optimizer settings for \"maxiter\" will be ignored')
        minimizer_settings.pop('maxiter')

if args.gpu_id is not None:
    template_settings['params']['gpu_id'] = {}
    template_settings['params']['gpu_id']['value'] = args.gpu_id
    template_settings['params']['gpu_id']['fixed'] = True

# Get the parameters
orig_params = template_settings['params'].copy()
hypo_params = hypo_settings['params'].copy()

# Make sure that theta23 is fixed:
logging.warn("Ensuring that theta23 is fixed for this analysis")
for key, value in orig_params.items():
  if 'theta23' in key:
    orig_params[key]['fixed'] = True
#print "params: ",params.items()

with Timer() as t:
    template_maker = TemplateMaker(get_values(orig_params),**template_settings['binning'])
profile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

data_types = [('data_NMH',True)]#, ('data_IMH',False)]

results = {}
# Store for future checking:
results['template_settings'] = template_settings
results['grid_settings'] = grid_settings

for data_tag, data_normal in data_types:
  results[data_tag] = {}

  data_params = select_hierarchy(orig_params, normal_hierarchy=data_normal)
  asimov_data_set = get_asimov_fmap(template_maker, get_values(data_params),
                                    chan=channel)
  results[data_tag]['asimov_data'] = asimov_data_set
  hypo_types = [('hypo_NMH',True)]#,('hypo_IMH',False)]
  for hypo_tag, hypo_normal in hypo_types:

    hypo_params = select_hierarchy(orig_params, normal_hierarchy=hypo_normal)
    # Calculate steps for theta23
    t23_params = { 'theta23': hypo_params['theta23'] }
    calc_steps( t23_params, grid_settings['steps'] )

    # Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ step for step in t23_params['theta23']['steps'] ]

    print "steplist: ",steplist
    # Prepare to store all the steps
    steps = { key:[] for key in t23_params.keys() }
    steps['chisquare'] = []

    # Iterate over the cartesian product, and set fixed parameter to value
    for step in steplist:
      print "running at %.2f"%step
      # be cautious...
      hypo_params_new = {}
      for (k, v) in hypo_params.items():
	hypo_params_new[k]=v.copy()
        if k=='theta23':
	  hypo_params_new[k]['value'] = step
	  hypo_params_new[k]['fixed'] = True

      steps['theta23'].append(step)

      if minimizer_settings is not None:
        with Timer() as t:
          chi2_data = find_min_chisquare_bfgs(asimov_data_set, template_maker, hypo_params_new,
                                              minimizer_settings, args.save_steps,
                                              normal_hierarchy=hypo_normal)
        profile.info("==> elapsed time for optimizer: %s sec"%t.secs)

      else: # Make use of the pull method
	# First, we need the template settings for truth and hypo.
	# Truth is always the same, but hypo changes according to
	# the value of the param. that is scanned.
	fisher, pulls = get_fisher_matrices(template_settings, true_nmh = data_normal, true_imh = not data_normal,
	hypo_nmh = hypo_normal, hypo_imh = not hypo_normal, take_finite_diffs = True, return_pulls = True,
	hypo_settings = hypo_params_new, template_maker = template_maker)

	min_chi2_vals = { }
	for p in pulls[data_tag][hypo_tag]:
	  min_chi2_vals[p[0]] = hypo_params_new[p[0]]['value'] + p[1]

	min_chi2_params_new = { }
	for (k, v) in hypo_params_new.items():
	  min_chi2_params_new[k] = v.copy()
	  if k in min_chi2_vals.keys():
	    min_chi2_params_new[k]['value'] = min_chi2_vals[k]

	min_chi2_hypo_data_set = get_asimov_fmap(template_maker, get_values(min_chi2_params_new),
                                    chan=channel)
	chi2_data = {}
	chi2_data['chisquare'] = [get_binwise_chisquare(asimov_data_set, min_chi2_hypo_data_set)]
		
	
      steps['chisquare'].append(chi2_data['chisquare'][-1])

      # Then save the minimized free params later??
      #print "\n\nsteps: ",steps
      results[data_tag][hypo_tag] = steps

logging.warn("FINISHED. Saving to file: %s"%args.outfile)
to_json(results,args.outfile)
