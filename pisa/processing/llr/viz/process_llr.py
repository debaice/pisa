#! /usr/bin/env python
#
# author: Timothy C. Arlen
#         tca3@psu.edu
#
# date:   3 July 2015
#
#

import os
from copy import deepcopy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'custom' 
mpl.rcParams['mathtext.rm'] = 'Arial'

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from tabulate import tabulate

from pisa.utils.hdf import from_hdf
from pisa.utils.log import logging, set_verbosity
from pisa.utils.params import get_values, select_hierarchy

from dfUtils import get_llr_data_frames, get_llh_ratios, show_frame
from plotUtils import plot_llr_distribution, plot_asimov_line, plot_fill
from plotUtils import make_scatter_plot, plot_posterior_params

from smartFormat import texLP

def get_false_h_best_params(llh_data):
    params = {}
    for tkey, ttag in [('true_NH', True), ('true_IH', False)]:
        try:
            params[tkey] = \
            deepcopy(llh_data[tkey]['false_h_best_fit']['false_h_settings'])
        except:
            params[tkey] = \
            deepcopy(get_values(select_hierarchy(
                llh_data['template_settings']['params'], ttag))
            )
    return params


def displayStats(mc_table):
    print ""
    print tabulate(
        mc_table,
        headers=['mcTrue', 'TH Mean', 'AH Mean', 'Count pval',
                 'Count sigma', 'Count sigma 2-sided',
                 'Gauss pval', 'Gauss sigma', 'Gauss sigma 2-sided'],
        tablefmt='grid'
    )


def make_llr_only_true_h(llr_dict, nbins, xlim=15):
    fig = plt.figure(figsize=(9, 6))

    mc_table = []
    colors = ['r', 'b']
    for ii, tkey in enumerate(['true_NH', 'true_IH']):
        opp_key = 'true_IH' if tkey == 'true_NH' else 'true_NH'
        label=tkey
        hvals, bincen, gfit = plot_llr_distribution(
            llr_dict[tkey], tkey, args.nbins, color=colors[ii], label=label)
        max_line = max(hvals)
        mean_llr = llr_dict[opp_key].mean()
        vline = plt.vlines(
            mean_llr, 0.1, max_line , colors='k')

        mcrow = plot_fill(
            llr_dict[opp_key], opp_key, mean_llr, hvals, bincen, gfit,
            alpha=0.5, hatch='xx', facecolor='black')
        plt.legend(framealpha=0.5, loc='upper right')
        plt.xlim(-xlim, xlim)
        ylim = plt.ylim()
        plt.ylim(0.8, ylim[-1])
        plt.yscale('log')
        mc_table.append(mcrow)

    plt.tight_layout()
    displayStats(mc_table)

    return fig


def set_xlim(llr_true_h, llr_false_h):
    ax = plt.gca()

    xmin =0; xmax = 0
    true_h_median = llr_true_h.median()
    false_h_median = llr_false_h.median()
    true_h_std = llr_true_h.std()
    false_h_std = llr_false_h.std()
    if(true_h_median < false_h_median):
        xmin = true_h_median - 4.5*true_h_std
        xmax = false_h_median + 4.5*false_h_std
    else:
        xmin = true_h_median + 4.5*true_h_std
        xmax = false_h_median - 4.5*false_h_std

    ax.set_xlim([xmin, xmax])
    return ax


def make_llr_with_false_h(llr_true_h, llr_false_h, nbins, xlim=15):
    fig = plt.figure(figsize=(15, 7))

    # 0) Plot true_h distributions, and get mean llr value
    logging.info(
        "  -->Plotting, calculating gaussian parameters for MC True:"
    )
    colors = ['b', 'g']
    for ii, tkey in enumerate(['true_NH', 'true_IH']):
        if tkey == 'true_NH':
            inj_ord = 'normal'
            hypo_ord = inj_ord
        else:
            inj_ord = 'inverted'
            hypo_ord = inj_ord
        ax = plt.subplot(1, 2, ii+1)
        ax.set_title(r'Injected $\nu$-mass ordering: %s' %inj_ord)
        ax.set_xlabel('Log-likelihood ratio')
        #label = r'$\mathcal{L}$( %s | IMH)/$\mathcal{L}$( %s | NMH)'%(tkey, tkey)
        #label = 'LLR(true Normal)' if tkey == 'true_NH' else 'LLR(true Inverted)'
        hvals, bincen, gfit = plot_llr_distribution(
            llr_cur=llr_true_h[tkey], tkey=tkey, bins=nbins,
            ax=ax, color=colors[ii], #label=label
        )
        t = ax.text(
            0.02, 0.85, "Hypothesized\nordering: %s" %hypo_ord,
            ha='left', va='top', transform=ax.transAxes,
            #bbox={'facecolor':'slategrey', 'alpha':0.5, 'pad':2}
        )

    logging.info("  -->Plotting  for false hierarchy best fit:")
    mc_table = []
    colors = ['b', 'g']
    for ii, tkey in enumerate(['true_NH', 'true_IH']):
        if tkey == 'true_NH':
            hypo_ord = 'inverted'
        else:
            hypo_ord = 'normal'

        ax = plt.subplot(1, 2, ii+1)
        label=r'H$_0$: Other Hierarchy'
        hvals, bincen, gfit = plot_llr_distribution(
            llr_cur=llr_false_h[tkey], tkey=tkey, bins=nbins,
            ax=ax, color=colors[ii], label=label
        )
        max_line = max(hvals)*1.2
        label=("Asimov_%s"%tkey)

        asimov_llr = llr_true_h[tkey].median()

        vline = plt.vlines(
            asimov_llr, 0.1, max_line , colors='k'
        )
        plt.text(
            x=asimov_llr, y=max_line*0.80, s='median',
            rotation=90, ha='right', va='top'
        )

        mcrow = plot_fill(
            llr_false_h[tkey], tkey, asimov_llr, hvals, bincen, gfit,
            alpha=0.5, hatch='xx', facecolor='black'
        )
        t = ax.text(
            0.98, 0.85, "Hypothesized\nordering: %s" %hypo_ord,
            ha='right', va='top', transform=ax.transAxes,
            #bbox={'facecolor':'slategrey', 'alpha':0.5, 'pad':2}
        )
        ax.text(
            0.28, 0.98,
            r"Gauss sig, 2-sided: %.2f" %mcrow[-1] +
            "\n" + 
            r"Count sig, 2-sided: %.2f" %mcrow[-4] +
            "\n" +
            r"LLR trials: $%s$" %texLP(len(llr_true_h[tkey])),
            ha='right', va='top', transform=ax.transAxes,
            bbox={'facecolor':'slategrey', 'alpha':0.5, 'pad':2}
        )
        plt.legend(framealpha=0.5, loc='upper right')

        ax = set_xlim(llr_true_h[tkey], llr_false_h[tkey])
        ax.set_yscale('log')
        ylim = ax.get_ylim()
        ax.set_ylim(0.8, max_line*1.2)
        mc_table.append(mcrow)
        plt.grid(b=True, which='minor', ls='-', color=[0.7]*3, lw=0.5, zorder=-5)
        plt.grid(b=True, which='major', ls='-', color=[0.5]*3, lw=0.5, zorder=-5)

    plt.tight_layout()
    displayStats(mc_table)
    return fig


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--llh-file', type=str,
                    help='''Processed LLH files to analyze''')
parser.add_argument('--nbins', type=int, default=50,
                    help='''Number of bins in x axis.''')
parser.add_argument('--no-false-h', action='store_true', default=False,
                    help='''Do not plot information about false_h llr dist''')
parser.add_argument('--xlim', type=float, default=15,
                    help='''Adjust xlimit to (-xlim, xlim) ''')

# Posterior parameter arguments
parser.add_argument('--params', action='store_true', default=False,
                    help="Plot all posterior parameter information in detail.")
parser.add_argument('--pbins', type=int, default=20,
                    help="Number of bins in x axis for posteriors.")
parser.add_argument('--scatter', metavar='PARAM_NAMES', type=str, nargs='+',
                    help='''Makes scatter plot for first two names listed
                    here''')
parser.add_argument('--plot-llh', action='store_true', default=False,
                    help='''Plot llh distribution with other parameters.''')
parser.add_argument('--false-h', action='store_true', default=False,
                    help='''Plot the false_h_best_fit posteriors rather than
                    the true_h_fiducial ones by default.''')

parser.add_argument('-s', '--save-fig', action='store_true', default=False,
                    help='Save all figures')
parser.add_argument('-v', '--verbose', action='count', default=0,
                    help='set verbosity level')

args = parser.parse_args()
set_verbosity(args.verbose)

# Configure plot settings
sns.set_context("poster") #if args.present else sns.set_context("talk")
sns.set_style("white")

llh_data = from_hdf(args.llh_file)
df_true_h, df_false_h = get_llr_data_frames(llh_data)
template_params = llh_data['template_settings']['params']
# If the best Asimov WH params have been fit for, need the best
# fit vals for posterior plotting
false_h_best_params = get_false_h_best_params(llh_data)

if args.verbose > 1:
    show_frame(df_true_h)

print "\n  columns: ", df_true_h[0].columns

################################################################
### 1) Plot LLR Distributions
################################################################
# df_true_h MUST be filled, but df_false_h is allowed to be empty
llr_dict_true_h = get_llh_ratios(df_true_h)
if (len(df_false_h) == 0 or args.no_false_h):
    logging.warn("No false hierarchy best fit llr distributions...")
    fig = make_llr_only_true_h(llr_dict_true_h, args.nbins, args.xlim)
else:
    logging.warn("Making llr distributions with false hierarchy best fit.")
    llr_dict_false_h = get_llh_ratios(df_false_h)
    fig = make_llr_with_false_h(llr_dict_true_h, llr_dict_false_h, args.nbins, args.xlim)


################################################################
### 2) Plot Posterior Distributions
################################################################
if args.save_fig:
    #filestem=args.llh_file.split('/')[-1]
    #basename = os.path.basename(args.llh_file) #args.llh_file.split('/')[-1]
    #filename=(filestem.split('.')[0]+'_LLR.png')
    rootname, extension = os.path.splitext(args.llh_file)
    llr_rootname = rootname + '_llr'
    logging.warn('Saving to file: %s' %llr_rootname+'.png')
    plt.savefig(llr_rootname+'.png', dpi=150)
    logging.warn('Saving to file: %s' %llr_rootname+'.pdf')
    plt.savefig(llr_rootname+'.pdf')

if args.params:
    df = df_false_h if args.false_h else df_true_h

    # Plot true_h_fiducial:
    figs, fignames = plot_posterior_params(
        df, template_params, plot_param_info=True,
        save_fig=args.save_fig, pbins=args.pbins,
        plot_llh=args.plot_llh, mctrue=not args.false_h,
        false_h_inj=false_h_best_params)

    if args.save_fig:
        for i, name in enumerate(fignames):
            figs[i].savefig(name, dpi=160)

else:
    plt.show()
