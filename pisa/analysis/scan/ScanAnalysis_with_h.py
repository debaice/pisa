#! /usr/bin/env python
#
# ScanAnalysis.py
#
# Runs a generic grid scan (either minimise or sample),
# using either a chi-square or LLH approach.
#
# author: Thomas Ehrhardt - tehrhard@uni-mainz.de
#

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json,to_json
from pisa.analysis.stats.Maps import get_asimov_fmap, flatten_map
from pisa.analysis.scan.Scan_2 import find_max_grid, find_min_chisquare_bfgs
from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.utils.params import get_values, select_hierarchy

def ScanAnalysis(template_settings, grid_settings=None, minimizer_settings=None, pseudo_data_settings=None,
		 IMH=True, NMH=True, scan_h = False, use_chisquare=True, use_llh=False, save_steps=True):
    '''
    Main function that runs the generic scan / minimization process for the chosen true
    hierarchies.
    '''
    do_grid_scan = False
    minimize = False
    if grid_settings is not None and minimizer_settings is not None:
        logging.warn('Currently, only either scan or minimization is possible, not both. Fall-back to scan for now!')
	do_grid_scan = True
	scan_settings = grid_settings
    elif not (grid_settings is None and minimizer_settings is None):
        if grid_settings is not None:
            scan_settings  = grid_settings
            do_grid_scan = True
        if minimizer_settings is not None:
            scan_settings = minimizer_settings
            minimize = True
            # Workaround for old scipy versions
            import scipy
            if scipy.__version__ < '0.12.0':
                logging.warn('Detected scipy version %s < 0.12.0'%scipy.__version__)
                if 'maxiter' in minimizer_settings:
                    logging.warn('Optimizer settings for \"maxiter\" will be ignored')
                    minimizer_settings.pop('maxiter')
    else:
	error_msg= 'You must specify either scan or minimizer settings. Aborting!'
        raise RuntimeError(error_msg)

    pd_settings = pseudo_data_settings if pseudo_data_settings is not None else template_settings
    
    # Make sure that both pseudo data and template are using the same channel.
    channel = template_settings['params']['channel']['value']
    if channel != pd_settings['params']['channel']['value']:
        error_msg = "Both template and pseudo data must have same channel!\n"
        error_msg += " pseudo_data_settings chan: '%s', template chan: '%s' "%(pd_settings['params']['channel']['value'],channel)
        raise ValueError(error_msg)

    tm_params = template_settings['params']
    tm_bins = template_settings['binning']
    pd_params = pd_settings['params']
    pd_bins = pd_settings['binning']
    if scan_h:
        # Artifically add the hierarchy parameter to the list of parameters
        # The method below will know how to deal with it
        tm_params['hierarchy_nh'] = { "value": 1., "range": [0.,1.],
                                      "fixed": False, "prior": None}
        tm_params['hierarchy_ih'] = { "value": 0., "range": [0.,1.],
                                      "fixed": False, "prior": None}

    # Initialise template maker(s)
    tm = TemplateMaker(get_values(tm_params),**tm_bins)

    pd_tm = TemplateMaker(get_values(pd_params),**pd_bins) if pseudo_data_settings is not None else tm

    # Account for true data
    chosen_data = []
    if IMH:
        chosen_data.append(('data_IMH',False))

    if NMH:
        chosen_data.append(('data_NMH',True))

    # //////////////////////////////////////////////////////////////////////
    # Generate two Asimov experiments (one for each true hierarchy),
    # and for each find the best matching template in each of the
    # hierarchy hypotheses.
    # //////////////////////////////////////////////////////////////////////
    results = {}
    for data_tag, data_normal in chosen_data:

        results[data_tag] = {}
        # Get "Asimov" average fiducial template for this hierarchy
	# Careful, if have defined deltam31 positive because it is also scanned over... still need to get template for deltam31<0 if data_IMH
	param_vals_data = get_values(select_hierarchy(pd_params, normal_hierarchy=data_normal))
	if not data_normal and param_vals_data['deltam31'] > 0.:
	    param_vals_data['deltam31'] = -param_vals_data['deltam31']
        asimov_fmap = flatten_map(pd_tm.get_template(param_vals_data
                                                     ),
				  channel
				  )
        # Minimize/Sample chi-square by comparing the templates from each of the hierarchies
	# to the fiducial Asimov template.
	if data_normal:
	    hypos = [('hypo_IMH', False)]
	else:
	    hypos = [('hypo_NMH', True)]
        for hypo_tag, hypo_normal in hypos:#[('hypo_NMH', True)], ('hypo_IMH', False)]:

	    if minimize:
		# This ignores whether llh or chi-square metric is requested.
                physics.info("Finding best fit for %s under %s."%(data_tag,hypo_tag))
	        profile.info("start minimising")
                grid_data = find_min_chisquare_bfgs(asimov_fmap, tm, tm_params,
                                                    scan_settings, save_steps, normal_hierarchy=hypo_normal)
                profile.info("stop minimising")

            if do_grid_scan:
                physics.info("Performing grid scan for %s under %s."%(data_tag,hypo_tag))
	        profile.info("start scan")
	        grid_data = find_max_grid(asimov_fmap, tm, tm_params,
                                          scan_settings, save_steps, hypo_normal)
                profile.info("stop scan")

            # Store the llh/chi-square data (all steps or only best fit)
            results[data_tag][hypo_tag] = grid_data

    output = {'results' : results,
              'template_settings' : template_settings,
              'scan_settings' : scan_settings}

    if pseudo_data_settings is not None:
        output['pseudo_data_settings'] = pseudo_data_settings

    return output


if __name__ == '__main__':
    parser = ArgumentParser(description='''Either samples or minimises varying a number of systematic parameters
    defined in the template settings file and saves the likelihood / chi-square values for all
    combinations of hierarchies.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t','--template_settings', type=str,
                        metavar='JSONFILE', required = True,
                        help='Settings related to the template generation and systematics.')

    parser.add_argument('-m','--minimizer_settings', type=str,
                        metavar='JSONFILE', default=None,
		        help='''Settings for the optimizer. Optimization can only be done
			     in conjunction with a chi-square metric.''')

    parser.add_argument('-g','--grid_settings', type=str,
                        metavar='JSONFILE', default=None,
                        help='''Settings for the grid sampling. Sampling can be done with
			     both metrics.''')

    parser.add_argument('-nh','--normal-truth', action='store_true',
		        default=False, dest='normal_truth',
		        help='Assume true normal hierarchy.')

    parser.add_argument('-ih','--inverted-truth', action='store_true',
		        default=False, dest='inverted_truth',
		        help='Assume true inverted hierarchy.')

    parser.add_argument('--scan_h', action='store_true',
                        default=False, dest='scan_h',
                        help='''Perform scan over the hierarchy parameter,
                             continuously extrapolating between the two hierarchies.''')
    """
    select_metric = parser.add_mutually_exclusive_group(required=False)

    select_metric.add_argument('-c','--chi-square', action='store_true', default=False,
		               dest='use_chisquare', help='Use chi-square metric.')
    select_metric.add_argument('-l','--llh', action='store_true', default=False,
		               dest='use_llh', help='Use log-likelihood metric (only with sampling).')
    """
    parser.add_argument('-pd','--pseudo_data_settings', type=str,
                        metavar='JSONFILE', default=None,
                        help='Settings for pseudo data templates, if desired to be different from template settings.')

    parser.add_argument('-s','--save-steps', action='store_true', default=False,
                        dest='save_steps', help="Save all steps the optimizer takes.")

    parser.add_argument('-o','--outfile', type=str, default='grid_data.json', metavar='JSONFILE',
                        help='Output filename.')

    parser.add_argument('-v', '--verbose', action='count', default=None,
                        help='set verbosity level')
    args = parser.parse_args()

    set_verbosity(args.verbose)

    #Read in the settings
    template_settings = from_json(args.template_settings)
    grid_settings = from_json(args.grid_settings) if args.grid_settings is not None else None
    minimizer_settings = from_json(args.minimizer_settings) if args.minimizer_settings is not None else None
    pseudo_data_settings = from_json(args.pseudo_data_settings) if args.pseudo_data_settings is not None else None

    #Perform the analysis
    grid_data = ScanAnalysis(template_settings, grid_settings, minimizer_settings, pseudo_data_settings,
                             IMH = args.inverted_truth, NMH = args.normal_truth, scan_h = args.scan_h, save_steps = args.save_steps)

    #Write results to file
    to_json(grid_data,args.outfile)


