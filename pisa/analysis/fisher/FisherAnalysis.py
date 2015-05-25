#! /usr/bin/env python
#
# FisherAnalysis.py
#
# Runs the Fisher Analysis method
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#	  Thomas Ehrhardt - tehrhard@uni-mainz.de

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import tempfile
import os
from itertools import product

from pisa.utils.log import logging, profile, physics, set_verbosity
from pisa.utils.jsons import from_json, to_json
from pisa.utils.params import select_hierarchy, get_free_params, get_values, fix_non_atm_params
from pisa.utils.utils import Timer
from pisa.utils.plot import delta_map

from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.analysis.stats.Maps import flatten_map

from pisa.analysis.fisher.gradients import get_gradients, get_hierarchy_gradients
from pisa.analysis.fisher.BuildFisherMatrix import build_fisher_matrix
from pisa.analysis.fisher.Fisher import FisherMatrix


def calculate_pulls(fisher, fid_maps_truth, fid_maps_hypo, gradient_maps, true_normal, hypo_normal):
  truth = 'data_NMH' if true_normal else 'data_IMH'
  hypo = 'hypo_NMH' if hypo_normal else 'hypo_IMH'
  d = []
  for chan_idx, chan in enumerate(fisher[truth][hypo]):
    if chan!='comb':
      chan_d = []
      f = fisher[truth][hypo][chan]
      if 'hierarchy' in f.parameters:
	f.removeParameter('hierarchy')
      # binwise derivatives w.r.t all parameters in this chan
      gm = gradient_maps[truth][hypo][chan]
      # binwise differences between truth and model in this chan
      # [d_bin1, d_bin2, ..., d_bin780]
      dm = np.array(delta_map(fid_maps_truth[chan], fid_maps_hypo[chan])['map']).flatten()
      # binwise statist. uncertainties for truth
      # [sigma_bin1, sigma_bin2, ..., sigma_bin3]
      sigma = np.array(fid_maps_truth[chan]['map']).flatten()
      for i, param in enumerate(f.parameters):
        chan_d.append([])
        assert(param in gm.keys())
        d_p_binwise = np.divide( np.multiply(dm, gm[param]['map']), sigma )
        # Sum over bins
        d_p = d_p_binwise.sum()
        chan_d[i] = d_p
      d.append(chan_d)
  # Binwise sum over (difference btw. fiducial maps times derivatives of
  # expected bin count / statistical uncertainty of bin count), summed over channels
  # Sum over channels (n-d vector, where n the number of systematics which are linearised)
  d = np.sum(d, axis=0)

  # This only needs to be multiplied by the (overall) Fisher matrix inverse.
  f_comb = fisher[truth][hypo]['comb']
  if 'hierarchy' in f_comb.parameters:
    f_comb.removeParameter('hierarchy')
  f_comb.calculateCovariance()
  pulls = np.dot(f_comb.covariance, d)
  return [ (pname, pull) for pname, pull in zip(f_comb.parameters, pulls.flat) ]


def get_fisher_matrices(template_settings, grid_settings, minimizer_settings=None, separate_fit=False, true_nmh=False, true_imh=True,
			hypo_nmh=True, hypo_imh=False, take_finite_diffs=False, return_pulls=False, dump_all_stages=False, save_templates=False, outdir=None):
  '''
  Main function that runs the Fisher analysis for the chosen true - assumed hierarchy combinations.

  Parameters
  -----------
  * template settings:	        dict: dictionary of settings for the template generation (required)
  * grid_settings:	        dict: dictionary of settings specifying the number of test points for each parameter (required)
  * minimizer_settings:		dict: dictionary of settings for the minimizer. If specified, a minimizer will determine
				the best fit point among the set of parameters in the alternative hierarchy, which
				will then be used as the fiducial model for the Fisher matrix.
  * separate_fit:		bool: given channel='all' and minimizer settings are specified, will enforce separate fits
				for the parameters in the alternative hierarchy for each final stage channel, thus allowing for
				different fiducial models for the corresponding Fisher matrices
  * true_nmh/true_imh:		bool: set to 'True' to compute the case of true normal (nmh) / inverted (imh, default) hierarchy
  * hypo_nmh/hypo_imh:		bool: set to 'True' to compute the case where hierarchy is identified as normal (default) / inverted
  * dump_all_stages:		bool: set to 'True' to save the expected templates at all stages for the different fiducial models
  * save_templates:		bool: set to 'True' to save the final stage bin counts for the different test points for each
				truth - hypo combination.
  * outdir:			str: directory to which to write the results

  Returns
  -----------
  * fisher: a dictionary of Fisher matrices, in the format (really depending on the truth - hypo combinations chosen):

  {'IMH': { 'NMH': {
		   'cscd': [...],
		   'trck': [...],
		   'comb': [...],
		   },
	    'IMH': {
                   'cscd': [...],
                   'trck': [...],
                   'comb': [...],
                   },
	  },
  'NMH': {...
         }
  }

  '''
  if outdir is None and (save_templates or dump_all_stages):
    logging.info("No output directory specified. Will save templates to current working directory.")
    outdir = os.getcwd()

  profile.info("start initializing")

  # Get the parameters
  orig_params = template_settings['params'].copy()
  bins = template_settings['binning'].copy()

  # Get the channels for which we are going to calculate a Fisher matrix
  orig_channel = orig_params['channel']['value']
  chans = []
  if orig_channel =='all':
    chans.append('cscd')
    chans.append('trck')
  elif orig_channel == 'no_pid':
    raise NotImplementedError('No particle-id not yet implemented! Aborting...')
  else:
    chans.append(orig_channel)
  # Artifically add the hierarchy parameter to the list of parameters.
  # The method get_hierarchy_gradients below will know how to deal with it.
  # Note: this parameter is only relevant for the mass hierarchy
  # sensitivity (hierarchy misidentification). In the case of correct
  # identification, we will remove it again.
  orig_params['hierarchy_nh'] = { "value": 1., "range": [0., 1.],
                                  "fixed": False, "prior": None}
  orig_params['hierarchy_ih'] = { "value": 0., "range": [0., 1.],
                                  "fixed": False, "prior": None}
  """
  # When fitting for the opposite hierarchy, keep all parameters
  # except theta23, which exhibits the largest bias, plus deltam31 fixed:
  alt_mh_fit_params = fix_non_atm_params(orig_params)
  """
  # Fit all parameters in the opposite hierarchy:
  alt_mh_fit_params = {}
  for key, value in orig_params.items():
    alt_mh_fit_params[key] = value.copy()
    if 'hierarchy' in key:
      alt_mh_fit_params[key]['fixed'] = True
  
  minimize = False

  chosen_data = []
  if true_imh:
    chosen_data.append(('data_IMH', False))
  if true_nmh:
    chosen_data.append(('data_NMH', True))
  if chosen_data == []:
    # In this case, only the fiducial maps (for both hierarchies) will be written
    logging.info("No Fisher matrices will be built.")
  else:
    hypos = []
    if ((true_imh and hypo_nmh) or (true_nmh and hypo_imh)):
      if minimizer_settings is None:
        logging.warn(""" A Fisher matrix is about to be built assuming hierarchy mis-id, but will """
                     """ be evaluated at the a-priori best fit values. Is this really what you want? """)
      else:
        minimize = True
    if hypo_imh:
      hypos.append(('hypo_IMH', False))
    if hypo_nmh:
      hypos.append(('hypo_NMH', True))
    if hypos == []:
      logging.info("No hierarchy hypothesis specified! Aborting.")
      return

  # Report on data - hypo combinations selected
  logging.info("Fisher matrices will be built for the combinations %s."%[("(truth %s, hypo %s)"%(t[0], h[0])) for (t, h) in product(chosen_data, hypos)])

  # There is no sense in performing any of the following steps if no Fisher matrices are to be built
  # and no templates are to be saved.
  if chosen_data!=[] or dump_all_stages or save_templates:

    # Initialise return dict to hold Fisher matrices, and also parameter maps
    fisher = { truth: { hypo: { } for hypo, hypo_normal in hypos } for truth, true_normal in chosen_data }
    pmaps = { truth: { hypo: { chan: {} for chan in chans } for hypo, hypo_normal in hypos} for truth, true_normal in chosen_data }
    gradient_maps = { truth: { hypo: { chan: {} for chan in chans } for hypo, hypo_normal in hypos} for truth, true_normal in chosen_data }
    pulls = { truth: { hypo: { } for hypo, hypo_normal in hypos} for truth, true_normal in chosen_data }
    # Get a template maker with the settings used to initialize
    template_maker = TemplateMaker(get_values(orig_params),**bins)

    profile.info("stop initializing\n")

    # Get_gradients and get_hierarchy_gradients will both (temporarily)
    # store the templates used to generate the gradient maps
    store_dir = outdir if save_templates else tempfile.gettempdir()

    # Calculate Fisher matrices for the user-defined cases (NHM true and/or IMH true)
    # TODO: if no optimisation is requested, then it does not matter whether data and hypothesis
    # coincide -> only need to run Fisher method twice!
    for ((truth, true_normal), (hypo, hypo_normal)) in product(chosen_data, hypos):
      # Note: new fiducial maps created for each truth - hypo combination
      fiducial_maps = {}
      logging.info("Running Fisher analysis for %s, %s."%(truth, hypo))
      logging.info("Generating fiducial templates for %s."%truth)
      fiducial_params = select_hierarchy(orig_params, normal_hierarchy=true_normal)

      profile.info("start template calculation")
      with Timer() as t:
        fid_maps = template_maker.get_template(get_values(fiducial_params),
                                               return_stages=dump_all_stages)
      profile.info("==> elapsed time for template: %s sec"%t.secs)

      fiducial_maps[truth] = fid_maps[4] if dump_all_stages else fid_maps
      # The fiducial model for the Fisher matrix depends on the channel considered
      fisher_eval_params = {c: {} for c in chans}

      if hypo_normal!=true_normal and minimize:
        max_data = {c: {} for c in chans}
        already_fit = False
        for chan in chans:
	  # Currently, channel determines which channel is going to be used for comparing the predictions
	  # to the data, and to get the best fit value.
	  # For 'all', do the same as for 'cscd' or 'trck', but then add the Fisher matrices together to get the
	  # final result.
	  # TODO: For no_pid, simply throw both channels together, and calculate the Fisher matrix.
	  if orig_channel == 'all' and separate_fit:
	    alt_mh_fit_params['channel']['value'] = chan

	  if not already_fit:
            # Get the best fit for the appropriate channel, determined by channel passed to flatten_map
            asimov_fmap = flatten_map(fiducial_maps[truth], chan=alt_mh_fit_params['channel']['value'])

	    logging.info("Finding best fit, looking at channel %s."%alt_mh_fit_params['channel']['value'])
	    profile.info("start minimising")

	    # Now optimise, alt_mh_fit_params have the channel information
	    # TODO: switch to find_min_chisquare_bfgs?
	    md = find_max_llh_bfgs(asimov_fmap, template_maker, alt_mh_fit_params,
					 minimizer_settings, save_steps=False, normal_hierarchy=hypo_normal,
					 check_octant=False)
	    profile.info("stop minimising")
	    md.pop('llh')

	    if not separate_fit:
	      already_fit = True

	  max_data[chan] = md
	  # Generate new alt. hierarchy Asimov data set from best fit values, which is then used
	  # as input to the Fisher matrix
	  fisher_eval_params[chan] = select_hierarchy(orig_params, normal_hierarchy=hypo_normal)
	  # First remove llh data, since we're only interested in the best fit values
	  # Define the fiducial model
	  for p in max_data[chan].keys():
	    fisher_eval_params[chan][p]['value'] = max_data[chan][p][0]

        logging.info("Generating fiducial templates after fitting %s."%hypo)

      else:
        # Either the assumed hierarchy corresponds to the injected one or minimisation wasn't requested.
        # In any case, we simply take the template settings as our fiducial model
        for chan in chans:
          fisher_eval_params[chan] = select_hierarchy(orig_params, hypo_normal)
          if hypo_normal==true_normal:
            # Remove the hierarchy parameter since we don't require it.
            del fisher_eval_params[chan]['hierarchy']

        logging.info("Generating fiducial templates for %s from global best fit"%hypo)
      
      # Generate fiducial template for hypothesis
      profile.info("start template calculation")
      # We can't just NOT generate the template for the channel that is not requested (or can we?),
      # so we are forced to drag that along with us for now
      fiducial_maps[hypo] = {c: {} for c in chans}
      for chan in chans:
        with Timer() as t:
          fid_maps = template_maker.get_template(get_values(fisher_eval_params[chan]),
                                                 return_stages=dump_all_stages)
        profile.info("==> elapsed time for template: %s sec"%t.secs)
        # Here, we're only interested in the final stage event numbers, which are our observables
        fiducial_maps[hypo][chan] = fid_maps[4][chan] if dump_all_stages else fid_maps[chan]
	# Note: 'params' one level below channel, not on the same as in get_template. Also,
	# select_hierarchy already applied, but the hierarchy info is in the hypo key anyway.
	fiducial_maps[hypo][chan]['params'] = fisher_eval_params[chan]

        # Get the free parameters (i.e. those for which the gradients should be calculated)
        free_params = get_free_params(fisher_eval_params[chan])
        for param in free_params.keys():
          # Special treatment for the hierarchy parameter
          if param=='hierarchy':
            # We don't need to pass hypo here, since hierarchy parameter only present
            # when data!=hypo anyway
            tpm, gm = get_hierarchy_gradients(truth,
					      chan,
					      fiducial_maps,
					      fisher_eval_params,
					      grid_settings,
					      store_dir)
          else:
            tpm, gm = get_gradients(truth,
				    hypo,
				    chan,
				    param,
				    template_maker,
				    fisher_eval_params,
				    grid_settings,
				    take_finite_diffs
				    )

	  pmaps[truth][hypo][chan][param] = tpm
	  gradient_maps[truth][hypo][chan][param] = gm

        logging.info("Building Fisher matrix for channel %s"%chan)

        # Build Fisher matrices for the given hierarchy.
        # Need to create best fit fiducial map first if optimisation performed
        if minimize:
          alt_mh_asimov_fmap = template_maker.get_template(get_values(fisher_eval_params[chan]), return_stages=False)
          fisher[truth][hypo][chan] = build_fisher_matrix(gradient_maps[truth][hypo][chan], alt_mh_asimov_fmap, fisher_eval_params[chan], chan)
        else:
          fisher[truth][hypo][chan] = build_fisher_matrix(gradient_maps[truth][hypo][chan], fiducial_maps['hypo_NMH'] if hypo_normal else fiducial_maps['hypo_IMH'], fisher_eval_params[chan], chan)

      # If Fisher matrices exist for both channels, add the matrices to obtain the combined one after we have created the individual ones.
      if len(fisher[truth][hypo].keys()) > 1:
        parameters = fisher[truth][hypo][fisher[truth][hypo].keys()[0]].parameters
        fisher[truth][hypo]['comb'] = FisherMatrix(matrix=np.array([f.matrix for f in fisher[truth][hypo].itervalues()]).sum(axis=0),
						   parameters=parameters,  #order is important here!
						   # use best_fit from one of the channels for the time being
						   best_fits=[fisher_eval_params[chans[0]][par]['value'] for par in parameters],
						   priors=[fisher_eval_params[chans[0]][par]['prior'] for par in parameters],
						   )
      if return_pulls:
	pull = calculate_pulls(fisher, fiducial_maps[truth], fiducial_maps[hypo], gradient_maps, true_normal, hypo_normal)
	pulls[truth][hypo] = pull

    if store_dir != tempfile.gettempdir():
      logging.info("Storing parameter maps.")
      to_json(pmaps, os.path.join(store_dir, "pmaps.json"))
      to_json(gradient_maps, os.path.join(store_dir, "gmaps.json"))

    return fisher, pulls

  else:
    logging.info("Nothing to be done.")
    return {}


if __name__ == '__main__':
  parser = ArgumentParser(description='''Runs the Fisher analysis method by varying a number of systematic parameters
                        defined in a settings.json file, taking the number of test points from a grid_settings.json file,
                        and saves the Fisher matrices for the selected true-hypo hierarchy combinations. If desired,
                        runs a minimizer first to locate the likelihood maximum when the hierarchy is misidentified.''',
                        formatter_class=ArgumentDefaultsHelpFormatter)

  parser.add_argument('-t','--template_settings', type=str,
                    metavar='JSONFILE', required=True,
                    help='Settings related to template generation and systematics.')

  parser.add_argument('-g','--grid_settings', type=str,
                    metavar='JSONFILE', required=True,
                    help='''Settings for the grid on which the gradients are
                    calculated (number of test points for each parameter).''')

  parser.add_argument('-m','--minimizer_settings', type=str,
                    metavar='JSONFILE', required=False,
                    help='''The Fisher matrix should be evaluated at
                    the likelihood (posterior) maximum. Given hierarchy misidentification,
                    this maximum needs to be found first. These are the settings
                    for the optimizer (if desired).''')

  parser.add_argument('--separate-fit', action='store_true',
		     default=False, dest='fit_sep',
		     help='''Enforces a separate fit of the alternative hierarchy parameters
		     in each analysis channel (if channel='all'), so that each has its unique
		     fiducial model at which the corresponding Fisher matrix is evaluated. Will
		     only take effect if minimizer settings are provided.''')

  parser.add_argument('--true-nmh', action='store_true',
		    default=False, dest='true_nmh',
		    help='Compute Fisher matrix for true normal hierarchy.')

  parser.add_argument('--true-imh', action='store_true',
                    default=False, dest='true_imh',
                    help='Compute Fisher matrix for true inverted hierarchy.')

  parser.add_argument('--hypo-nmh', action='store_true',
		    default=False, dest='hypo_nmh',
		    help='Assume hierarchy is identified as normal.')

  parser.add_argument('--hypo-imh', action='store_true',
		    default=False, dest='hypo_imh',
		    help='Assume hierarchy is identified as inverted.')
  parser.add_argument('--finite-diffs', action='store_true',
		    default=False, dest='finite_diffs',
		    help='''Interpolate linearly between the parameter bounds instead of fitting.
		    Might reduce precision for the least linear systematics.''')
  parser.add_argument('--return-pulls', action='store_true',
		    default=False, dest='return_pulls',
		    help='''Calculate and return the pull of each systematic.''')
  parser.add_argument('-d','--dump-all-stages', action='store_true', dest='dump_all_stages', default=False,
                    help='''Store histograms at all simulation stages for fiducial model in 
                    normal and inverted hierarchy.''')

  sselect = parser.add_mutually_exclusive_group(required=False)

  sselect.add_argument('-s','--save-templates', action='store_true',
                    default=False, dest='save_templates',
                    help='Save all the templates used to obtain the gradients.')

  sselect.add_argument('-n','--no-save-templates', action='store_false',
                    default=True, dest='save_templates',
                    help='Do not save the templates for the different test points.')

  parser.add_argument('-o', '--outdir', type=str, default=os.getcwd(), metavar='DIR',
                    help='Output directory')

  parser.add_argument('-v', '--verbose', action='count', default=None,
                    help='Set verbosity level.')

  args = parser.parse_args()

  # Set verbosity level
  set_verbosity(args.verbose)

  # Read the template settings
  template_settings = from_json(args.template_settings)

  # This file only contains the number of test points for each parameter (and perhaps eventually a non-linearity criterion)
  grid_settings  = from_json(args.grid_settings)

  # Read the minimizer settings if given
  minimizer_settings = from_json(args.minimizer_settings) if args.minimizer_settings is not None else None

  # Get the Fisher matrices for the desired true vs. assumed hierarchy combinations and fiducial settings
  fisher_matrices, pulls = get_fisher_matrices(template_settings, grid_settings, minimizer_settings, args.fit_sep,
                                        args.true_nmh, args.true_imh, args.hypo_nmh, args.hypo_imh, args.finite_diffs,
					args.return_pulls, args.dump_all_stages, args.save_templates, args.outdir)

  # Fisher matrices are saved in any case
  for truth in fisher_matrices:
    for hypo in fisher_matrices[truth]:
      fisher_basename='fisher_%s_%s'%(truth, hypo)
      for chan in fisher_matrices[truth][hypo]:
        if chan == 'comb':
          outfile = os.path.join(args.outdir, fisher_basename+'.json')
          logging.info("%s, %s: writing combined Fisher matrix to %s"%(truth, hypo, outfile))
        else:
          outfile = os.path.join(args.outdir,fisher_basename+'_%s.json'%chan)
          logging.info("%s, %s: writing Fisher matrix for channel %s to %s"%(truth, hypo, chan, outfile))
        # Save matrix for each truth - hypo - channel combination in a separate file
        fisher_matrices[truth][hypo][chan].saveFile(outfile)
