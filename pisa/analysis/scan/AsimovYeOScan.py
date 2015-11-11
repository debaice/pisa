#! /usr/bin/env python
#
# AsimovYeOScan.py
#
# Scans YeO over a  grid. Uses Asimov method.  
#
# author: Debanjan Bose - debanjan.tifr@gmail.com
#
# date:   11-Nov-2015
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
parser.add_argument('-o','--outfile',type=str,default='llh_data.json',metavar='JSONFILE',
                    help='''Output filename.''')
parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='''set verbosity level''')
args = parser.parse_args()

set_verbosity(args.verbose)

#Read in the settings
template_settings = from_json(args.template_settings)
hypo_settings = from_json(args.template_settings)
grid_settings = from_json(args.grid_settings)

channel = template_settings['params']['channel']['value']

if args.gpu_id is not None:
  template_settings['params']['gpu_id'] = {}
  template_settings['params']['gpu_id']['value'] = args.gpu_id
  template_settings['params']['gpu_id']['fixed'] = True

# Get the parameters, but make sure to keep template settings untouched
orig_params = deepcopy(template_settings['params'])

# Check whether parameter exists at all (for both hierarchies)
if not args.scan_par in select_hierarchy(template_settings['params'], True) or not args.scan_par in select_hierarchy(template_settings['params'], False):
  raise RuntimeError("Parameter %s not found. Aborting!"%args.scan_par)


with Timer() as t:
    template_maker = TemplateMaker(get_values(orig_params),**template_settings['binning'])
tprofile.info("==> elapsed time to initialize templates: %s sec"%t.secs)

data_types = [('data_NMH',True), ('data_IMH',False)]

results = {}
# Store for future checking:
results['template_settings'] = template_settings

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
    scan_params = { args.scan_par: hypo_params[args.scan_par] }
    calc_steps( scan_params, grid_settings['steps'] )

    # Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [(name, step) for step in scan_params[name]['steps']] for name in sorted(scan_params.keys()) ]

    # Prepare to store all the steps, chisquare values and best fits
    steps = { key:[] for key in scan_params.keys() }
    steps['chisquare'] = []

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


	# Note: The parameters that are scanned need to have "fixed" set to false in template settings!
        # For all other parameters "fixed" should be set to true in the template settings. 
      priors = { param: prior for (param, prior) in 
				    zip (sorted(get_free_params(select_hierarchy(template_settings['params'], hypo_normal)).keys()),
					 get_param_priors(get_free_params(select_hierarchy(template_settings['params'], hypo_normal))))
	       }
      chi2_data = {}
      best_fits = []
      hypo_data_set = get_asimov_fmap(template_maker, get_values(hypo_params_new),
                                  channel=channel)


      # print get_values(hypo_params_new)

      chi2 = get_binwise_chisquare(asimov_data_set, hypo_data_set)
      chi2 += sum([ priors[param].chi2(hypo_params_new[param]['value'])
                      for param in priors.keys() ])

      chi2_data['chisquare'] = [chi2]


      steps['chisquare'].append(chi2_data['chisquare'][-1])

	
      results[data_tag][hypo_tag] = steps
      #	Let's not lose everything in case something goes wrong.
      to_json(results, args.outfile)
logging.warn("FINISHED. Saving to file: %s"%args.outfile)
to_json(results, args.outfile)
