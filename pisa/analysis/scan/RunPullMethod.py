#! /usr/bin/env python
#
# RunPullMethod.py
#
# Scans a 1- or 2-d parameter grid,
# analytically minimising the chisquare over the other - linearised - parameters
# by calculating the Fisher matrix at each point and determining the pulls.
# Alternatively, can also make use of numerical minimisation at each point, which is
# slower by a factor of on the order of 10. In addition to the chisquare, also returns
# the best fits of all parameters which are optimised.
#
# author: Thomas Ehrhardt - tehrhard@uni-mainz.de
#
# date:   25-May-2015
#

import sys
import numpy as np
from itertools import product
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy

from pisa.analysis.llr.LLHAnalysis import find_opt_scipy
from pisa.analysis.stats.LLHStatistics import get_binwise_chisquare
from pisa.analysis.stats.Maps import get_asimov_fmap
from pisa.analysis.scan.Scan import calc_steps
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.analysis.fisher.FisherAnalysis import get_fisher_matrices

from pisa.utils.log import logging, tprofile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import get_values, select_hierarchy, fix_atm_params, get_atm_params, get_free_params, get_param_priors#, get_prior_chisquare
from pisa.utils.utils import Timer

parser = ArgumentParser(
    description='''Pull method.''',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-t','--template_settings',type=str,
                    metavar='JSONFILE', required = True,
                    help='''Settings related to the template generation and systematics.''')
parser.add_argument('-p','--param-to-scan', type=str, required=True, dest='scan_par',
		    help='''Scan this parameter.''')
parser.add_argument('-p2','--2nd-parameter', type=str, required=False, dest='scan_par2',
		    help='''Scan a 2-d grid, including this parameter.''')
parser.add_argument('-m','--minimizer_settings',type=str,
                    metavar='JSONFILE', required = False,
                    help='''Settings related to the optimizer used in the LLR analysis.''')
parser.add_argument('-g','--grid_settings',type=str,metavar='JSONFILE', required = True,
                    help='''Get chisquare at defined grid values,
                    according to these input settings.''')
group = parser.add_mutually_exclusive_group()
group.add_argument('--only-correct-h', action='store_true', default=False,
		    dest='only_correct_h', help='''Only assume hierarchy is correctly identified.''')
group.add_argument('--only-cross-h', action='store_true', default=False,
		    dest='only_x_h', help='''Only assume hierarchy is mis-identified.''')
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
if minimizer_settings is not None:
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

# Get the parameters, but make sure to keep template settings untouched
orig_params = deepcopy(template_settings['params'])

# Check whether parameter exists at all (for both hierarchies)
if not args.scan_par in select_hierarchy(template_settings['params'], True) or not args.scan_par in select_hierarchy(template_settings['params'], False):
  raise RuntimeError("Parameter %s not found. Aborting!"%args.scan_par)

# Make sure that parameter to scan is fixed (+ possibly 2nd parameter):
logging.warn("Ensuring that %s is fixed for this analysis."%args.scan_par)

if args.scan_par2 is not None:
  logging.warn("Ensuring that %s is fixed for this analysis."%args.scan_par2)
  for key, value in orig_params.items():
    if args.scan_par in key or args.scan_par2 in key:
      orig_params[key]['fixed'] = True

else:
  for key, value in orig_params.items():
    if args.scan_par in key:
      orig_params[key]['fixed'] = True

with Timer() as t:
    template_maker = TemplateMaker(get_values(orig_params),**template_settings['binning'])
tprofile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

data_types = [('data_NMH',True), ('data_IMH',False)]

results = {}
# Store for future checking:
results['template_settings'] = template_settings
results['grid_settings'] = grid_settings
if minimizer_settings is not None:
  results['minimizer_settings'] = minimizer_settings

global_opt_flags = []
for data_tag, data_normal in data_types:
  results[data_tag] = {}

  data_params = select_hierarchy(template_settings['params'], normal_hierarchy=data_normal)
  asimov_data_set = get_asimov_fmap(template_maker, get_values(data_params),
                                    channel=channel)

  results[data_tag]['asimov_data'] = asimov_data_set

  hypo_types = [('hypo_NMH',True),('hypo_IMH',False)]
  for hypo_tag, hypo_normal in hypo_types:
    if args.only_x_h:
	if hypo_normal==data_normal:
	  continue
    if args.only_correct_h:
	if hypo_normal!=data_normal:
	  continue

    # The parameters that are to be scanned will be fixed in here:
    hypo_params = select_hierarchy(orig_params, normal_hierarchy=hypo_normal)

    # Calculate steps for parameters
    if args.scan_par2 is not None:
      scan_params = { args.scan_par: hypo_params[args.scan_par], args.scan_par2: hypo_params[args.scan_par2] }
    else:
      scan_params = { args.scan_par: hypo_params[args.scan_par] }

    calc_steps( scan_params, grid_settings['steps'] )

    # Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [(name, step) for step in scan_params[name]['steps']] for name in sorted(scan_params.keys()) ]

    # Prepare to store all the steps, chisquare values and best fits
    steps = { key:[] for key in scan_params.keys() }
    steps['chisquare'] = []
    best_fits = { key:[] for key in get_free_params(data_params)  }

    # Iterate over the cartesian product, and set fixed parameter to value
    for pos in product(*steplist):
      print "running at ",pos
      # be cautious...
      hypo_params_new = {}
      for (k, v) in hypo_params.items():
	hypo_params_new[k]=v.copy()
	# now adjust the values of the parameters to be scanned
	if k in scan_params.keys():
	  param_val = pos[sorted(scan_params.keys()).index(k)][1]
	  print "fixing %s to %.2f"%(k,param_val)
	  hypo_params_new[k]['value'] = param_val
	  steps[k].append(param_val)

      if len(get_free_params(hypo_params_new).keys())==0:
	# Note: The parameters that are scanned need to have "fixed" set to false in template settings!
	priors = { param: prior for (param, prior) in 
					zip (sorted(get_free_params(select_hierarchy(template_settings['params'], hypo_normal)).keys()),
					     get_param_priors(get_free_params(select_hierarchy(template_settings['params'], hypo_normal))))
		 }
	chi2_data = {}
	best_fits = []
	hypo_data_set = get_asimov_fmap(template_maker, get_values(hypo_params_new),
                                    channel=channel)

	chi2 = get_binwise_chisquare(asimov_data_set, hypo_data_set)
	chi2 += sum([ priors[param].chi2(hypo_params_new[param]['value'])
                      for param in priors.keys() ])

        chi2_data['chisquare'] = [chi2]

      elif minimizer_settings is not None:
        with Timer() as t:
          chi2_data, dict_flags = find_opt_scipy(asimov_data_set, template_maker, hypo_params_new,
                                    minimizer_settings, args.save_steps,
                                    normal_hierarchy=hypo_normal, check_octant=True,
				    metric_name='chisquare')
        tprofile.info("==> elapsed time for optimizer: %s sec"%t.secs)
	global_opt_flags.append(dict_flags)

	# store the best fit values
	for p in chi2_data:
	  if p!='chisquare':
	  #if p!='llh':
	    best = chi2_data[p][0]
	    best_fits[p].append(best)

      else: # Make use of the pull method
	# First, we need the template settings for truth and hypo.
	# Truth is always the same, but hypo changes according to
	# the value of the param. that is scanned.
	fisher, pulls = get_fisher_matrices(template_settings, true_nmh = data_normal, true_imh = not data_normal,
	hypo_nmh = hypo_normal, hypo_imh = not hypo_normal, take_finite_diffs = True, return_pulls = True,
	hypo_settings = hypo_params_new, template_maker = template_maker)

	min_chi2_vals = { }
	for p in pulls[data_tag][hypo_tag]:
	  best = hypo_params_new[p[0]]['value'] + p[1]
	  min_chi2_vals[p[0]] = best
          best_fits[p[0]].append(best)

	# artificially add the parameter(s) scanned over to this dictionary, so
	# we can add the prior penalty terms for these

	for scanned_par in scan_params:
	  min_chi2_vals[scanned_par] = hypo_params_new[scanned_par]['value']

	min_chi2_params_new = { }
	for (k, v) in hypo_params_new.items():
	  min_chi2_params_new[k] = v.copy()
	  if k in min_chi2_vals.keys():
	    min_chi2_params_new[k]['value'] = min_chi2_vals[k]

	min_chi2_hypo_data_set = get_asimov_fmap(template_maker, get_values(min_chi2_params_new),
                                    channel=channel)
	chi2_data = {}
	chi2 = get_binwise_chisquare(asimov_data_set, min_chi2_hypo_data_set)
	priors = get_param_priors(get_free_params(select_hierarchy(template_settings['params'],hypo_normal)))
	#print priors
	#print sorted(min_chi2_vals.items()), priors
	chi2 += sum([ prior.chi2(opt_val)
		      for (opt_val, prior) in zip([ opt for (p, opt) in sorted(min_chi2_vals.items())], priors) ])
	chi2_data['chisquare'] = [chi2]

      steps['chisquare'].append(chi2_data['chisquare'][-1])
      #steps['chisquare'].append(chi2_data['llh'][-1])
      steps['best_fits'] = best_fits
	
      results[data_tag][hypo_tag] = steps
      #	Let's not lose everything in case something goes wrong.
      to_json(results, args.outfile)
logging.warn("FINISHED. Saving to file: %s"%args.outfile)
to_json(results, args.outfile)
#print global_opt_flags
#to_json(global_opt_flags, "opt_dict_flags.json")
