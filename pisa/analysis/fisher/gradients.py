#! /usr/bin/env python
#
# Gradients.py
#
# Tools for calculating the gradients.
#
# author: Lukas Schulte - schulte@physik.uni-bonn.de
#         Sebastian Boeser - sboeser@uni-mainz.de
#	  Thomas Ehrhardt - tehrhard@uni-mainz.de

import numpy as np
import os
import tempfile
import copy

from pisa.utils.jsons import to_json
from pisa.utils.log import logging, profile
from pisa.utils.params import get_values
from pisa.utils.utils import Timer

from pisa.analysis.stats.Maps import flatten_map



def derivative_from_polycoefficients(coeff, loc):
    """
    Return derivative of a polynomial of the form

        f(x) = coeff[0] + coeff[1]*x + coeff[2]*x**2 + ...

    at x = loc
    """

    result = 0.

    for n in range(len(coeff))[1:]: # runs from 1 to len(coeff)

        result += n*coeff[n]*loc**(n-1)

    return result


def get_derivative_map(data, chan, fiducial=None , degree=2):
  """
  Get the approximate derivative of data w.r.t parameter par
  at location loc with polynomic degree of approximation, default: 2.

  Data is a dictionary of the form
  {
  'test_point1': {'params': {},
		 {'map': [[],[],...],
			    'ebins': [],
			    'czbins': []
			  },
		  }

  'test_point2': ...
  }
  """
  derivative_map = {}
  test_points = sorted(data.keys())

  # Flatten data map for use with polyfit
  channel_data = [ np.array(data[pvalue]['map']).flatten() for pvalue in test_points ]
  # Polynomial fit of bin counts
  channel_fit_params = np.polyfit(test_points, channel_data, deg=degree)
  # Get partial derivatives at fiducial values
  derivative_map['map'] = derivative_from_polycoefficients(channel_fit_params[::-1], fiducial['value'])

  return derivative_map


def get_steps(param, grid_settings, fiducial_params):
  """
  Prepare the linear sequence of test points: use a globally valid
  number of test points if grid_settings makes no specifications
  for the parameter.
  """
  try:
    n_points = grid_settings['npoints'][param]
  except:
    n_points = grid_settings['npoints']['default']

  return np.linspace(fiducial_params[param]['range'][0],fiducial_params[param]['range'][1],n_points)



def get_hierarchy_gradients(data_tag, chan, fiducial_maps, fiducial_params, grid_settings, store_dir):
  """
  Use the hierarchy interpolation between the two fiducial maps to obtain the
  gradients.
  """
  logging.info("Working on parameter hierarchy.")

  # The ranges are the same even if the channel differs, so get them from one of the channels
  # (or the only one present)
  steps = get_steps('hierarchy', grid_settings, fiducial_params[chan])

  hmap = {step:{} for step in steps}
  for h in steps:
    # Superpose bin counts (fiducial maps for different channels within same hierarchy
    # possibly generated assuming different parameter values)
    if data_tag == 'data_NMH':
      hmap[h]['map'] = fiducial_maps['NMH'][chan]['map']*h + fiducial_maps['IMH'][chan][chan]['map']*(1.-h)
      # Obtain binning from one of the maps, since identical by construction (cf. FisherAnalysis)
      hmap[h]['ebins'] = fiducial_maps['NMH'][chan]['ebins']
      hmap[h]['czbins'] = fiducial_maps['NMH'][chan]['czbins']
    elif data_tag == 'data_IMH':
      hmap[h]['map'] = fiducial_maps['NMH'][chan][chan]['map']*h + fiducial_maps['IMH'][chan]['map']*(1.-h)
      # Obtain binning from one of the maps, since identical by construction (cf. FisherAnalysis)
      hmap[h]['ebins'] = fiducial_maps['IMH'][chan]['ebins']
      hmap[h]['czbins'] = fiducial_maps['IMH'][chan]['czbins']


  # TODO: give hmap the same structure as pmaps?
  # Get_derivative_map works even if 'params' and 'ebins','czbins' not in 'data'

  gradient_map = get_derivative_map(hmap, chan, fiducial_params[chan]['hierarchy'], degree=2)

  return gradient_map



def get_gradients(data_tag, hypo_tag, chan, param, template_maker, fiducial_params, grid_settings, store_dir):
  """
  Use the template maker to create all the templates needed to obtain the gradients.
  """
  logging.info("Working on parameter %s."%param)

  steps = get_steps(param, grid_settings, fiducial_params[chan])

  pmaps = {}

  # Generate one template for each value of the parameter in question and store in pmaps
  for param_value in steps:

      # Make the template corresponding to the current value of the parameter
      with Timer() as t:
          maps = template_maker.get_template(
              get_values(dict(fiducial_params[chan],**{param:dict(fiducial_params[chan][param],
                                                            **{'value': float(param_value)})})))
      profile.info("==> elapsed time for template: %s sec"%t.secs)
      pmaps[param_value] = maps[chan]

  # TODO: Reduce number of output files
  # Store the maps used to calculate partial derivatives
  if store_dir != tempfile.gettempdir():
  	logging.info("Writing maps for parameter %s to %s"%(param,store_dir))

  to_json(pmaps, os.path.join(store_dir, param+"_"+data_tag+"_"+hypo_tag+"_"+chan+".json"))

  gradient_map = get_derivative_map(pmaps, chan, fiducial_params[chan][param], degree=2)

  return gradient_map

def check_param_linearity(pmaps, prange=None, chan="no_pid", plot_hist=False, param="", plot_for_energy_bin=-1):
  """
  Take the templates generated from different values of a systematic and produce
  a nonlinearity distribution, by calculating a binwise nonlinearity based on
  a linear fit to the bin entries.

  Parameters
  -----------
  * pmaps: 			dictionary of parameter values and corresponding maps with bin counts;
				as written out by FisherAnalysis.py (cf. get_derivative_map)
  * prange: 			1-d list; determines which test points should be considered, i.e. only those within
				(for the Fisher method, it is only important that each parameter be approx.
				linear within the range corresponding to the accuracy of the experiment)
  * channel: 			PID channel string; 'cscd', 'trck' or 'no_pid' (sum)
  * plot_hist:  		if 'true', plots nonlinearity distribution in a 1-d histogram
  * param:			name of the parameter examined; used for histogram title
  * plot_for_energy_bin:        if > 0, will return plots of bin counts vs. parameter value for all
				coszen bins given this energy bin

  Returns
  -----------
  * nonlinearities: 1-d array of nonlinearity in each bin in (E, coszen)
  """

  if prange is None:
    prange = [-np.inf, np.inf]

  if len(pmaps.keys())<3:
    raise RuntimeError("It seems less than n=3 points were evaluated! Consider rerunning the Fisher analysis with n>2 and then come back here.")

  test_points = np.array(sorted([ float(val) for val in pmaps.keys() ]))
  
  """
  # For old maps
  # Fet binning information, which might be needed later
  ebins = pmaps.values()[0]['cscd']['ebins'] if 'cscd' in pmaps.values()[0] else pmaps.values()[0]['trck']['ebins']
  czbins = pmaps.values()[0]['cscd']['czbins'] if 'cscd' in pmaps.values()[0] else pmaps.values()[0]['trck']['czbins']
  nebins, nczbins = len(ebins)-1, len(czbins)-1
  logging.info("%s energy and %s coszen bins detected..."%(nebins, nczbins))
  """

  # For new maps
  ebins = pmaps.values()[0]['ebins']
  czbins = pmaps.values()[0]['czbins']
  nebins, nczbins = len(ebins)-1, len(czbins)-1
  logging.info("%s energy and %s coszen bins detected..."%(nebins, nczbins))

  npoints = len(test_points)

  npoints_in_range = 0
  for val in test_points:
    if val >= prange[0] and val <= prange[1]:
      npoints_in_range+=1

  if npoints_in_range < 3:
    prange = [test_points[npoints/2-1], test_points[npoints/2+1]]
    logging.warn("Range contains less than 3 of the points probed. Expanded to %s."%prange)

  test_points_in_range = np.array([val for val in test_points if val >= prange[0] and val <= prange[1] ])
  # Keep the strings for accessing the corresponding dict entries in pmaps
  test_points_in_range_str = np.array(sorted([val for val in pmaps.keys() if float(val) >= prange[0] and float(val) <= prange[1] ]))

  # Pre-shape pmaps for use with polyfit
  # For old maps:
  # bin_counts_data = np.array([ flatten_map(pmaps[val], chan=chan) for val in test_points_in_range_str ])
  # For new maps:
  bin_counts_data = np.array([ pmaps[val]['map'].flatten() for val in test_points_in_range_str ])

  # Perform a linear fit
  linfit_params = np.polyfit(test_points_in_range, bin_counts_data, deg=1)

  # Plot for first energy bin if plot_for_energy_bin = 1
  bin_range_to_plot = range(0)
  if plot_for_energy_bin > 0:
    import matplotlib.pyplot as plt
    if plot_for_energy_bin <= nebins:
      bin_range_to_plot = range((plot_for_energy_bin-1)*nczbins, nczbins*plot_for_energy_bin)

  # Create an array of fit values which has the same shape as the data
  bin_counts_fit = []
  for (val_idx, val) in enumerate(test_points_in_range):
    bin_counts_fit.append([])
    for (bin_idx, counts) in enumerate(bin_counts_data[val_idx]):
      linfit = np.polyval(linfit_params[:, bin_idx], val)
      logging.trace("test point %.2f, bin %s: data %s vs. lin. fit %s"%(val, bin_idx, counts, linfit))
      bin_counts_fit[val_idx].append(linfit)        
  bin_counts_fit = np.array(bin_counts_fit)

  # Calculate relative squared deviations of data from linear fits
  delta_data_fit = np.power(np.divide(np.subtract(bin_counts_data, bin_counts_fit), bin_counts_data), 2)

  # Now determine the binwise contribution from each test point and normalise
  # (reuse bin_idx, which after the last for-loop corresponds to the total number of bins)
  nonlinearities = np.divide([ np.sum(delta_data_fit[:, i]) for i in xrange(bin_idx+1) ], len(test_points_in_range))

  # Generate plots for all coszen bins if energy bin was specified
  if bin_range_to_plot != range(0):
    figsize = (14, 10)
    cols = 4
    fig, axes = plt.subplots(figsize=figsize, nrows=len(bin_range_to_plot) // cols, ncols=cols, sharex=False, dpi=100)
    fig.suptitle('%s,'%param + ' channel: %s,'%chan + r' $E_{\mathrm{reco}} \, \in \, [%.2f,\,%.2f]\, \mathrm{GeV}$'%(ebins[plot_for_energy_bin-1], ebins[plot_for_energy_bin]), size=16)
    x = test_points_in_range
    for i in bin_range_to_plot:
      chisquare = nonlinearities[i]
      row = ((i % nczbins) // cols)
      col = (i % nczbins) % cols
      axes[row, col].scatter(x, bin_counts_data[:, i], marker='o', s=4, label='data', color='firebrick')
      axes[row, col].plot(x, bin_counts_fit[:, i], ls='-', label='lin. fit', c='firebrick')
      plt.setp(axes[row, col], title=r'$\cos(\theta_{\mathrm{reco}}) \, \in \, [%.2f,\, %.2f]$'%(czbins[i%nczbins], czbins[(i%nczbins)+1]))
      axes[row, col].legend(loc='lower right', fontsize=8, scatterpoints=1)
      axes[row, col].annotate(r'$\chi^2_{\mathrm{fit}}=%s$'%chisquare, (0.05, 0.85), xycoords='axes fraction', textcoords='axes fraction', size=10, color='dodgerblue')
      axes[row, col].grid()
      axes[row, col].set_xlim(min(x), max(x))
    fig.text(0.5, 0.03, param, fontsize=14, ha='center', va='center')
    fig.text(0.02, 0.5, 'bin counts', fontsize=14, ha='center', va='center', rotation='vertical')
    fig.tight_layout()
    fig.subplots_adjust(left=0.06, bottom=0.08, top=0.9)
    plt.show()

  if plot_hist:
    # 1d & 2d histograms of nonlinearities
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4), dpi=100)
    plt.hist(nonlinearities, histtype='step', bins=20, color='k')
    title = "Linearity"
    if param is not None: title += " of %s"%param
    title += " (range: [%.5f, %.5f])"%(min(test_points_in_range), max(test_points_in_range))
    title += " , channel: %s"%chan
    plt.title(title, fontsize=10)
    plt.xlabel(r"$\chi^2/n_{\mathrm{points}}$")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()

    # Also show nonlinearity per bin on grid in coszen, energy
    nonl_reshaped = nonlinearities.reshape(ebins.shape[0]-1, czbins.shape[0]-1)
    e, z = np.meshgrid(ebins, czbins)
    plt.figure(figsize=(14,10), dpi=100)
    if param!="":
      plt.title('nonlinearity map for %s,'%param + ' channel: %s'%chan, fontsize=16)
    else:
      plt.title('nonlinearity map, '+'channel: %s'%chan, fontsize=16)
    plt.pcolor(z.T, e.T, nonl_reshaped)
    c = plt.colorbar()
    c.set_label(r'$\chi^2$', fontsize=14)
    plt.xlabel(r'$\cos(\theta_{\mathrm{reco}})$', fontsize=16)
    plt.ylabel(r'$E_{\mathrm{reco}}\, [\mathrm{GeV}]$', fontsize=16)
    plt.ylim(min(ebins), max(ebins))
    plt.show()

  return nonlinearities
