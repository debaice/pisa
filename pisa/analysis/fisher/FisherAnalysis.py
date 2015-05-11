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
from pisa.utils.jsons import from_json,to_json
from pisa.utils.params import select_hierarchy, get_free_params, get_values, fix_non_atm_params
from pisa.utils.utils import Timer

from pisa.analysis.llr.LLHAnalysis import find_max_llh_bfgs
from pisa.analysis.TemplateMaker import TemplateMaker
from pisa.analysis.stats.Maps import flatten_map

from pisa.analysis.fisher.gradients import get_gradients, get_hierarchy_gradients
from pisa.analysis.fisher.BuildFisherMatrix import build_fisher_matrix
from pisa.analysis.fisher.Fisher import FisherMatrix


def get_fisher_matrices(template_settings, grid_settings, minimizer_settings=None, true_nmh=False, true_imh=True, hypo_nmh=True, hypo_imh=False,
                        dump_all_stages=False, save_templates=False, outdir=None):
  '''
  Main function that runs the Fisher analysis for the chosen hierarchy(ies) (true inverted, hypothesis normal by default).

  Returns a dictionary of Fisher matrices, in the format:
  {'IMH': {'cscd': [...],
          'trck': [...],
          'comb': [...],
          },
  'NMH': {'cscd': [...],
          'trck': [...],
          'comb': [...],
         }
  }

  If save_templates=True and no hierarchy is given, only fiducial templates will be written out;
  if one is given, then the templates used to obtain the gradients will be written out in
  addition.
  '''
  if outdir is None and (save_templates or dump_all_stages):
    logging.info("No output directory specified. Will save templates to current working directory.")
    outdir = os.getcwd()

  profile.info("start initializing")

  # Get the parameters
  orig_params = template_settings['params'].copy()
  bins = template_settings['binning'].copy()
  # Artifically add the hierarchy parameter to the list of parameters.
  # The method get_hierarchy_gradients below will know how to deal with it.
  # Note: this parameter is only relevant for the mass hierarchy
  # sensitivity (hierarchy misidentification). In the case of correct
  # identification, we will remove it again.
  orig_params['hierarchy_nh'] = { "value": 1., "range": [0., 1.],
                                  "fixed": False, "prior": None}
  orig_params['hierarchy_ih'] = { "value": 0., "range": [0., 1.],
                                  "fixed": False, "prior": None}
  # When fitting for the opposite hierarchy, keep all parameters
  # except theta23, which exhibits the largest bias, plus deltam31 fixed.
  alt_mh_fit_params = fix_non_atm_params(orig_params)

  minimize = False

  chosen_data = []
  if true_imh:
    chosen_data.append(('IMH', False))
  if true_nmh:
    chosen_data.append(('NMH', True))
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
      hypos.append(('IMH', False))
    if hypo_nmh:
      hypos.append(('NMH', True))
    if hypos == []:
      logging.info("No hierarchy hypothesis specified! Aborting.")
      return

  # report on data/hypo combinations selected
  logging.info("Fisher matrices will be built for the combinations %s."%[("(truth %s, hypo %s)"%(t[0], h[0])) for (t, h) in product(chosen_data, hypos)])

  # There is no sense in performing any of the following steps if no Fisher matrices are to be built
  # and no templates are to be saved.
  if chosen_data!=[] or dump_all_stages or save_templates:

    # Initialise return dict to hold Fisher matrices
    fisher = { truth: { hypo: {'cscd':[], 'trck':[], 'comb':[]} for hypo, hypo_normal in hypos } for truth, true_normal in chosen_data }

    # Get a template maker with the settings used to initialize
    template_maker = TemplateMaker(get_values(orig_params),**bins)

    profile.info("stop initializing\n")

    # Generate fiducial templates for both hierarchies (needed for partial derivatives
    # w.r.t. hierarchy parameter)
    fiducial_maps = {}
    for hierarchy in ['NMH','IMH']:

      logging.info("Generating fiducial templates for %s."%hierarchy)

      # Get the fiducial parameter values corresponding to this hierarchy
      fiducial_params = select_hierarchy(orig_params, normal_hierarchy=(hierarchy=='NMH'))

      # Generate fiducial maps, either all of them or only the ultimate one
      profile.info("start template calculation")
      with Timer() as t:
        fid_maps = template_maker.get_template(get_values(fiducial_params),
                                               return_stages=dump_all_stages)
      profile.info("==> elapsed time for template: %s sec"%t.secs)

      fiducial_maps[hierarchy] = fid_maps[4] if dump_all_stages else fid_maps

      # save fiducial map(s)
      # all stages
      if dump_all_stages:
        stage_names = ("0_unoscillated_flux", "1_oscillated_flux", "2_oscillated_counts", "3_reco", "4_pid")
        stage_maps = {}
        for stage in xrange(0,len(fid_maps)):
          stage_maps[stage_names[stage]] = fid_maps[stage]
        logging.info("Writing fiducial maps (all stages) for %s to %s."%(hierarchy,outdir))
        to_json(stage_maps, os.path.join(outdir, "fid_map_"+hierarchy+".json"))
      # only the final stage
      elif save_templates:
        logging.info("Writing fiducial map (final stage) for %s to %s."%(hierarchy,outdir))
        to_json(fiducial_maps[hierarchy], os.path.join(outdir, "fid_map_"+hierarchy+".json"))

    # Get_gradients and get_hierarchy_gradients will both (temporarily)
    # store the templates used to generate the gradient maps
    store_dir = outdir if save_templates else tempfile.gettempdir()

    # Calculate Fisher matrices for the user-defined cases (NHM true and/or IMH true)
    # TODO: if no optimisation is requested, then it does not matter whether data and hypothesis
    # coincide -> only need to run Fisher method twice!
    for ((truth, true_normal), (hypo, hypo_normal)) in product(chosen_data, hypos):
      logging.info("Running Fisher analysis for true %s, hypo %s."%(truth, hypo))
      if hypo!=truth and minimize:
        asimov_fmap = flatten_map(fiducial_maps[truth], chan='all')
        logging.info("Finding best fit.")
        profile.info("start minimising")
        max_data = find_max_llh_bfgs(asimov_fmap, template_maker, alt_mh_fit_params,
                                     minimizer_settings, save_steps=False, normal_hierarchy=hypo_normal,
                                     check_octant=False)
        profile.info("stop minimising")
        # generate new alt. hierarchy fiducial Asimov data set from best fit values
        # fiducial model for Fisher matrix (PINGU best-fit for theta23, deltam31)
        fisher_eval_params = select_hierarchy(orig_params, normal_hierarchy=hypo_normal)
        max_data.pop('llh')
        for p in max_data.keys():
          fisher_eval_params[p]['value'] = max_data[p][0]
      else:
        # either the assumed hierarchy corresponds to the injected one or minimisation wasn't requested,
        # in any case, we simply take the template settings as our fiducial model
        fisher_eval_params = select_hierarchy(orig_params, hypo_normal)
        if hypo==truth:
          del fisher_eval_params['hierarchy']
        # The fiducial params are selected from the hierarchy case that does NOT match
        # the data, as we are varying from this model to find the 'best fit'
        # fiducial_params = select_hierarchy(params,not data_normal)

      # Get the free parameters (i.e. those for which the gradients should be calculated)
      # free_params = select_hierarchy(get_free_params(params),not data_normal)
      free_params = get_free_params(fisher_eval_params)
      gradient_maps = {}
      for param in free_params.keys():
        # Special treatment for the hierarchy parameter
        if param=='hierarchy':
          # we don't need to pass hypo here, since hierarchy parameter only present
          # when data!=hypo anyway
          gradient_maps[param] = get_hierarchy_gradients("data_"+truth,
                                                         fiducial_maps,
                                                         fisher_eval_params,
						         grid_settings,
						         store_dir,
						         )
        else:
          gradient_maps[param] = get_gradients("data_"+truth,
                                               "hypo_"+hypo,
                                               param,
                                               template_maker,
                                               fisher_eval_params,
                                               grid_settings,
                                               store_dir
                                               )

      logging.info("Building Fisher matrix.")

      # Build Fisher matrices for the given hierarchy
      # need to create best fit fiducial map first if optimisation performed
      if minimize:
        alt_mh_asimov_fmap = template_maker.get_template(get_values(fisher_eval_params), return_stages=False)
        fisher[truth][hypo] = build_fisher_matrix(gradient_maps, alt_mh_asimov_fmap, fisher_eval_params)
      else:
        fisher[truth][hypo] = build_fisher_matrix(gradient_maps, fiducial_maps['NMH'] if hypo_normal else fiducial_maps['IMH'], fisher_eval_params)

      # If Fisher matrices exist for both channels, add the matrices to obtain the combined one.
      if len(fisher[truth][hypo].keys()) > 1:
        fisher[truth][hypo]['comb'] = FisherMatrix(matrix=np.array([f.matrix for f in fisher[truth][hypo].itervalues()]).sum(axis=0),
                                                   parameters=gradient_maps.keys(),  #order is important here!
                                                   best_fits=[fisher_eval_params[par]['value'] for par in gradient_maps.keys()],
                                                   priors=[fisher_eval_params[par]['prior'] for par in gradient_maps.keys()],
                                                   )
    return fisher

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
  fisher_matrices = get_fisher_matrices(template_settings, grid_settings, minimizer_settings,
                                        args.true_nmh, args.true_imh, args.hypo_nmh, args.hypo_imh,
                                        args.dump_all_stages, args.save_templates, args.outdir)

  
  # Fisher matrices are saved in any case
  for truth in fisher_matrices:
    for hypo in fisher_matrices[truth]:
      fisher_basename='fisher_data_%s_hypo_%s'%(truth, hypo)
      for chan in fisher_matrices[truth][hypo]:
        if chan == 'comb':
          outfile = os.path.join(args.outdir, fisher_basename+'.json')
          logging.info("true %s, hypo %s: writing combined Fisher matrix to %s"%(truth, hypo, outfile))
        else:
          outfile = os.path.join(args.outdir,fisher_basename+'_%s.json'%chan)
          logging.info("true %s, hypo %s: writing Fisher matrix for channel %s to %s"%(truth, hypo, chan, outfile))
        fisher_matrices[truth][hypo][chan].saveFile(outfile)
