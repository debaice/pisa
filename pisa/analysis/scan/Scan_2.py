#
# Scan.py
#
# Helper module for brute-force scanning analysis over the whole parameter space
#
# author: Sebastian Boeser <sboeser@uni-mainz.de>
#

import sys
import numpy as np
import scipy.optimize as opt
from itertools import product

from pisa.utils.log import logging, physics, profile
from pisa.utils.params import get_values, select_hierarchy, get_fixed_params, get_free_params, get_prior_llh, get_prior_chisquare, get_param_values, get_param_scales, get_param_bounds, get_param_priors
from pisa.utils.utils import Timer
from pisa.analysis.stats.Maps import flatten_map
from pisa.analysis.stats.LLHStatistics import get_binwise_llh, get_binwise_chisquare

def calc_steps(params, settings):
    '''
    Get the actual grid values for each key. If settings is a list of
    values, use these directly as steps. If settings has single value,
    generate this many steps within the bounds given for the parameter.
    Parameters are identified by names, or "*" which is the default for all
    parameters
    '''

    #Collect the steps settings for each parameter
    for key in params:

        #If specific steps are given, use these
        if key in settings:
            params[key]['steps'] = settings[key]
        else:
            params[key]['steps'] = settings['*']

    #Now convert number of steps to actual steps
    for key in params:
        #ignore if we already have those
        if isinstance(params[key]['steps'],np.ndarray): continue

        #calculate the steps
        lower, upper = params[key]['range']
        nsteps = params[key]['steps']
        params[key]['steps'] = np.linspace(lower,upper,nsteps)

    #report for all
    for name, steps in [ (k,v['steps']) for k,v in params.items()]:
       logging.debug("Using %u steps for %s from %.5f to %.5f" %
                          (len(steps), name, steps[0], steps[-1]))

def mcmc(fmap, template_maker, params, grid_settings, save_steps=True, normal_hierarchy=True, use_chisquare=True, use_llh=False):
    '''MCMC sampling posterior'''
    import emcee

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    #Obtain just the priors
    priors = get_param_priors(free_params)
    
    ndim = len(free_params)

    #Calculate steps for all free parameters
    calc_steps(free_params, grid_settings['steps'])

    #Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [step for step in param['steps']] for name, param in sorted(free_params.items())]
    free_param_names = [ [name for step in param['steps']] for name, param in sorted(free_params.items())]
    p0_names = []
    for names in product(*free_param_names):
	p0_names.append([n for n in names])
    p0_names = p0_names[0]
    ndim = len(p0_names)
    p0 = []
    for pos in product(*steplist):
	p0.append([p for p in pos])
    nwalkers = np.array(p0).shape[0]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc_llh, args=[fixed_params, p0_names, priors, fmap, template_maker, params])

    pos, prob, state = sampler.run_mcmc(p0, 50)
    sampler.reset()
    
    sampler.run_mcmc(pos, 200)

    print "Mean acceptance fraction: %.3f"%np.mean(sampler.acceptance_fraction)
    print sampler.flatchain
    return sampler.flatchain 
    
def mcmc_llh(pos, fixed_params, pos_names, priors, fmap, template_maker, params):
    name_pos = [] 
    for i in xrange(0,len(pos)):
        name_pos.append((pos_names[i],pos[i]))
    print name_pos
    
    """
    for np in name_pos:
        if np[1] > params[np[0]]['range'][1] or np[1] < params[np[0]]['range'][0]:
            return -inf
    """
    template_params = dict(list(name_pos) + get_values(fixed_params).items())
    profile.info('start template calculation')
    true_template = template_maker.get_template(template_params)
    profile.info('stop template calculation')
    true_fmap = flatten_map(true_template)

    #get sorted vals to match with priors
    vals = [ v for k,v in sorted(name_pos) ]

    chisquare = -get_binwise_chisquare(fmap,true_fmap)
    chisquare -= sum([ get_prior_chisquare(vals,sigma,value) for (vals, (sigma,value)) in zip(vals,priors) ])

    #llh = get_binwise_llh(fmap,true_fmap)
    #llh += sum([ get_prior_llh(vals,sigma,value) for (vals,(sigma,value)) in zip(vals,priors)])
    print chisquare
    return chisquare

def find_max_grid(fmap, template_maker, params, grid_settings,
		  save_steps=True, normal_hierarchy=True, use_chisquare=True, use_llh=False):
    '''
    Finds the template (and free systematic params) that maximize
    likelihood that the data came from the chosen template of true
    params, using a brute force grid scan over the whole parameter space.

    returns a dictionary of chi-square and parameter values, in the format:
      {'chisquare': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params varied in the scan,
    and they hold a list of all the values tested, unless save_steps is False,
    in which case they are one element in length-the best fit params and
    minimum chi-square.
    '''

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    #Obtain just the priors
    priors = get_param_priors(free_params)

    #Calculate steps for all free parameters
    calc_steps(free_params, grid_settings['steps'])

    #Build a list from all parameters that holds a list of (name, step) tuples
    steplist = [ [(name,step) for step in param['steps']] for name, param in sorted(free_params.items())]
    #Prepare to store all the steps and llh/chi-square values
    steps = {}
    if use_llh:
	steps['llh'] = {key:[] for key in free_params.keys()}
        steps['llh']['values'] = []
    if use_chisquare:
        steps['chisquare'] = {key:[] for key in free_params.keys()}
	steps['chisquare']['values'] = []

    # List of steps for all parameters except hierarchy parameter. For the cartesian product, we need to create the 
    # normal and inverted hierarchy maps.
    # this is the correct steplist, even when the h param is not supposed to be scanned
    steplist_wo_h = [ [(name,step) for step in param['steps']] for name, param in sorted(free_params.items()) if name!='hierarchy']
    for pos in product(*steplist_wo_h):
        # make pos mutable so we can append the ('hierarchy',h) tuple later on
        pos_wo_h = list(pos)
        template_params = dict(pos_wo_h + get_values(fixed_params).items())
        for key in template_params.keys():
            try: print "  >>param: %s value: %s"%(key,str(template_params[key]['value']))
            except: continue
	print pos_wo_h
        # If we are evaluating the chisquare / llh for the hierarchy parameter, the
        # true template is the superposition of the normal and inverted templates
        # for the combination of the values of the other parameters that are varied.
        # Note: It's a waste to recalculate the templates when it's only the value
        # of the hierarchy parameter that changes...
        # This is where the generation of the "true" template differs, depending on whether
        # h in free params or not
        if 'hierarchy' in free_params.keys():
            true_hierarchy_maps = {}
            for hierarchy in ['NMH','IMH']: # create the true templates for both hierarchies, using the same value of the parameter that is scanned
                logging.info("Generating true templates for %s."%hierarchy)
		for name_param_comb in pos_wo_h:
		    if name_param_comb[0] != 'deltam31': # put positive values into the template settings file for deltam31
                        if (hierarchy=='NMH' and normal_hierarchy) or (hierarchy=='IMH' and not normal_hierarchy):
		            # we are creating the true template for the hierarchy that matches the 'hypothesis'. template_params are the correct ones
		            eval_params = template_params
                        else:
		            # we are creating the true template for the hierarchy that doesn't match the 'hypothesis'.
		            # we need to get the parameter values for this hierarchy
                            eval_params = get_values(select_hierarchy(params,normal_hierarchy=(hierarchy=='NMH')))
		            # now set the value that is scanned also in the opposite hierarchy
		            eval_params[name_param_comb[0]] = name_param_comb[1]
		    else: # we are dealing with deltam31
		        if (hierarchy=='NMH' and normal_hierarchy):
		            eval_params = template_params
		        elif (hierarchy=='NMH' and not normal_hierarchy):
			    eval_params = get_values(select_hierarchy(params,normal_hierarchy=True))
		        # use the same absolute value of deltam31 for both hierarchies
		        elif (hierarchy=='IMH' and normal_hierarchy):
			    eval_params = get_values(select_hierarchy(params,normal_hierarchy=False))
			    eval_params[name_param_comb[0]] = -name_param_comb[1]
		        elif (hierarchy=='IMH' and not normal_hierarchy):
			    eval_params = template_params
			    eval_params[name_param_comb[0]] = -name_param_comb[1]
			        
		#print eval_params
                profile.info("start true template calculation")
                with Timer() as t:
                    true_template = template_maker.get_template(eval_params)
                profile.info("==> elapsed time for template: %s sec"%t.secs)

                true_hierarchy_maps[hierarchy] = true_template
             
            for h in free_params['hierarchy']['steps']:
		# recreate pos for each value of h, so we don't append multiple h values
           	pos = list(pos_wo_h)
		pos.append(('hierarchy',h))
                template_params = dict(pos + get_values(fixed_params).items())
                channels = true_hierarchy_maps['NMH'].keys()
                true_template = {c: 0 for c in channels}
                for channel in channels:
                    if channel!='params':
		        # replace 'h' with correct value from pos...
                        true_template[channel] = {}
                        true_template[channel]['map'] = true_hierarchy_maps['NMH'][channel]['map']*h + true_hierarchy_maps['IMH'][channel]['map']*(1.-h)
		        true_template[channel]['ebins'] = true_hierarchy_maps['NMH'][channel]['ebins']
                        true_template[channel]['czbins'] = true_hierarchy_maps['NMH'][channel]['czbins']
            
                true_template['params'] = true_hierarchy_maps['NMH']['params'] if normal_hierarchy else true_hierarchy_maps['IMH']['params']

                true_fmap = flatten_map(true_template)
  
                # get sorted vals to match with priors
                vals = [ v for k,v in sorted(pos) ]

                if use_llh:
                    #and calculate the likelihood
                    llh = -get_binwise_llh(fmap,true_fmap)
                    llh -= sum([ get_prior_llh(vals,sigma,value) for (vals,(sigma,value)) in zip(vals,priors)])

                    # Save all values to steps and report
                    steps['llh']['values'].append(llh)
                    physics.debug("LLH is %.2f at: "%llh)

                    for key, val in pos:
                        steps['llh'][key].append(val)
                        physics.debug(" %20s = %6.4f" %(key, val))

                if use_chisquare:
                    chisquare = get_binwise_chisquare(fmap,true_fmap)
                    chisquare += sum([ get_prior_chisquare(vals,sigma,value) for (vals, (sigma,value)) in zip(vals,priors) ])

                    steps['chisquare']['values'].append(chisquare)
                    physics.debug("chi-square is %.2f at: "%chisquare)

                    for key, val in pos:
                        steps['chisquare'][key].append(val)
                        physics.debug(" %20s = %6.4f" %(key, val))

        else: # Hierarchy parameter not scanned / minimised over, proceed as usual:
            # Now get true template
            profile.info('start template calculation')
            true_template = template_maker.get_template(template_params)
            profile.info('stop template calculation')
            true_fmap = flatten_map(true_template)
            
            vals = [ v for k,v in sorted(pos) ]

            if use_llh:
                #and calculate the likelihood
                llh = -get_binwise_llh(fmap,true_fmap)
                llh -= sum([ get_prior_llh(vals,sigma,value) for (vals,(sigma,value)) in zip(vals,priors)])
                # Save all values to steps and report
                steps['llh']['values'].append(llh)
                physics.debug("LLH is %.2f at: "%llh)
                for key, val in pos:
                    steps['llh'][key].append(val)
                    physics.debug(" %20s = %6.4f" %(key, val))

            if use_chisquare:
                chisquare = get_binwise_chisquare(fmap,true_fmap)
                chisquare += sum([ get_prior_chisquare(vals,sigma,value) for (vals, (sigma,value)) in zip(vals,priors) ])

                steps['chisquare']['values'].append(chisquare)
                physics.debug("chi-square is %.2f at: "%chisquare)

                for key, val in pos:
                    steps['chisquare'][key].append(val)
                    physics.debug(" %20s = %6.4f" %(key, val))
           
    #Find best fit value
    if use_llh:
        max_llh = min(steps['llh']['values'])
        max_llh_pos = steps['llh']['values'].index(max_llh)
        #Report best fit
        physics.info('Found best LLH = %.2f in %d calls at:'
                 %(max_llh,len(steps['llh']['values'])))

	for name, vals in steps['llh'].items():
            physics.info('  %20s = %6.4f'%(name,vals[max_llh_pos]))

            #only save this maximum if asked for
            if not save_steps:
                steps['llh'][name]=vals[max_llh_pos]

    if use_chisquare:
        min_chisquare = min(steps['chisquare']['values'])
        min_chisquare_pos = steps['chisquare']['values'].index(min_chisquare)
	physics.info('Found minimum chi-square = %.2f in %d calls at:'
                 %(min_chisquare,len(steps['chisquare']['values'])))

        for name, vals in steps['chisquare'].items():
            physics.info('  %20s = %6.4f'%(name,vals[min_chisquare_pos]))

            #only save this maximum if asked for
            if not save_steps:
                steps['chisquare'][name]=vals[min_chisquare_pos]

    return steps

# Adapted from find_max_llh_bfgs from LLHAnalysis.py
def find_min_chisquare_bfgs(fmap,template_maker,params,bfgs_settings,save_steps=False,
                            normal_hierarchy=True):
    '''
    Finds the template (and free systematic params) that minimise
    delta chi-square under the normal/inverted hierarchy hypothesis
    w.r.t. fmap, using the limited memory BFGS algorithm subject to bounds
    (l_bfgs_b).

    returns a dictionary of chi-square and parameter values, in the format:
      {'chisquare': [...],
       'param1': [...],
       'param2': [...],
       ...}
    where 'param1', 'param2', ... are the free params varied by the
    optimizer, and they hold a list of all the values tested in
    optimizer algorithm, unless save_steps is False, in which case
    they are one element in length-the best fit params and minimum chi-square.
    '''

    # Get params dict which will be optimized (free_params) and which
    # won't be (fixed_params) but are still needed for get_template()
    fixed_params = get_fixed_params(select_hierarchy(params,normal_hierarchy))
    free_params = get_free_params(select_hierarchy(params,normal_hierarchy))

    init_vals = get_param_values(free_params)
    scales = get_param_scales(free_params)
    bounds = get_param_bounds(free_params)
    priors = get_param_priors(free_params)
    names  = sorted(free_params.keys())

    # Scale init-vals and bounds to work with bfgs opt:
    init_vals = np.array(init_vals)*np.array(scales)
    bounds = [bounds[i]*scales[i] for i in range(len(bounds))]

    opt_steps_dict = {key:[] for key in names}
    opt_steps_dict['chisquare'] = []

    const_args = (names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors)

    physics.info('%d parameters to be optimized'%len(free_params))
    for name,init,(down,up),(prior, best) in zip(names, init_vals, bounds, priors):
        physics.info(('%20s : init = %6.4f, bounds = [%6.4f,%6.4f], '
                     'best = %6.4f, prior = '+
                     ('%6.4f' if prior else "%s"))%
                     (name, init, up, down, best, prior))

    physics.debug("Optimizer settings:")
    for key,item in bfgs_settings.items():
        physics.debug("  %s -> `%s` = %.2e"%(item['desc'],key,item['value']))

    best_fit_vals,chisquare,dict_flags = opt.fmin_l_bfgs_b(chisquare_bfgs,
                                                     init_vals,
                                                     args=const_args,
                                                     approx_grad=True,
                                                     iprint=0,
                                                     bounds=bounds,
                                                     **get_values(bfgs_settings))

    best_fit_params = { name: value for name, value in zip(names, best_fit_vals) }

    #Report best fit
    physics.info('Found minimum chisquare = %.2f in %d calls at:'
        %(chisquare,dict_flags['funcalls']))
    for name, val in best_fit_params.items():
        physics.info('  %20s = %6.4f'%(name,val))

    #Report any warnings if there are
    lvl = logging.WARN if (dict_flags['warnflag'] != 0) else logging.DEBUG
    for name, val in dict_flags.items():
        physics.log(lvl," %s : %s"%(name,val))

    if not save_steps:
        # Do not store the extra history of opt steps:
        for key in opt_steps_dict.keys():
            opt_steps_dict[key] = [opt_steps_dict[key][-1]]

    return opt_steps_dict

# Adapted from llh_bfgs in LLHAnalysis.py
def chisquare_bfgs(opt_vals,*args):
    '''
    Function that the bfgs algorithm tries to minimize. Essentially,
    it is a wrapper function around get_template() and
    get_binwise_chisquare().

    This fuction is set up this way, because the fmin_l_bfgs_b
    algorithm must take a function with two inputs: params & *args,
    where 'params' are the actual VALUES to be varied, and must
    correspond to the limits in 'bounds', and 'args' are arguments
    which are not varied and optimized, but needed by the
    get_template() function here. Thus, we pass the arguments to this
    function as follows:

    --opt_vals: [param1,param2,...,paramN] - systematics varied in the optimization.
    --args: [names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors]
      where
        names: are the dict keys corresponding to param1, param2,...
        scales: the scales to be applied before passing to get_template
          [IMPORTANT! In the optimizer, all parameters must be ~ the same order.
          Here, we keep them between 0.1,1 so the "epsilon" step size will vary
          the parameters in roughly the same precision.]
        fmap: pseudo data flattened map
        fixed_params: dictionary of other paramters needed by the get_template()
          function
        template_maker: template maker object
        opt_steps_dict: dictionary recording information regarding the steps taken
          for each trial of the optimization process.
        priors: gaussian priors corresponding to opt_vals list.
          Format: [(prior1,best1),(prior2,best2),...,(priorN,bestN)]
    '''


    names,scales,fmap,fixed_params,template_maker,opt_steps_dict,priors = args

    # free parameters being "optimized" by minimizer re-scaled to their true values.
    unscaled_opt_vals = [opt_vals[i]/scales[i] for i in xrange(len(opt_vals))]

    unscaled_free_params = { names[i]: val for i,val in enumerate(unscaled_opt_vals) }
    template_params = dict(unscaled_free_params.items() + get_values(fixed_params).items())

    # Now get true template, and compute chi-square
    with Timer() as t:
        if template_params['theta23'] == 0.0:
            logging.info("Zero theta23, so generating no oscillations template...")
            true_template = template_maker.get_template_no_osc(template_params)
        else:
            true_template = template_maker.get_template(template_params)
    profile.info("==> elapsed time for template maker: %s sec"%t.secs)
    true_fmap = flatten_map(true_template,chan=template_params['channel'])

    chisquare = get_binwise_chisquare(fmap,true_fmap)
    chisquare += sum([ get_prior_chisquare(opt_val,sigma,value)
                 for (opt_val,(sigma,value)) in zip(unscaled_opt_vals,priors)])

    # Save all optimizer-tested values to opt_steps_dict, to see
    # optimizer history later
    for key in names:
        opt_steps_dict[key].append(template_params[key])
    opt_steps_dict['chisquare'].append(chisquare)

    physics.debug("chisquare is %.2f at: "%chisquare)
    for name, val in zip(names, opt_vals):
        physics.debug(" %20s = %6.4f" %(name,val))

    return chisquare
