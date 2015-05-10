#
# plots.py
#
# Utility function for plotting maps
#
# author: Sebastian Boeser
#         sboeser@physik.uni-bonn.de
#
# date:   2014-01-27

from utils import is_linear, is_logarithmic

import numpy as np
import matplotlib.pyplot as plt

def show_map(pmap, title=None, cbar = True,
             vmin=None, vmax=None,
             emin=None, emax=None,
             czmin=None, czmax=None,
             invalid=False, logE=None,
             log=False, fontsize=16,
             xlabel=r'cos $\vartheta_\mathrm{zenith}$',
             ylabel='Energy [GeV]',
             zlabel=None,
             zlabel_size='large',
             **kwargs):
    '''Plot the given map with proper axis labels using matplotlib.
       The axis orientation follows the PINGU convention:
          - energy to the right
          - cos(zenith) to the top

    Keyword arguments:

      * title: Show this title above the map

      * cbar: show a colorbar for the z-axis (True or False)

      * vmin/vmax: set the minimum/maximum color (z-axis) values.
                   If no value is provided, vmin and vmax are choosen
                   symmetrically to comprehend the full range.

      * emin/emax: set the minimum maximum energy range to show.
                   If not value is provided use the full range covered
                   by the axis.

      * czmin/czmax: same as above for cos(zenith)

      * log: use a logarithmic (log10) colour (z-axis) scale

      * logE: show the x-axis on a logarithmic scale (True or False)
              Default is "guessed" from the bins size.

      * invalid: if True show color values for NaN, None, -inf and +inf,
                 otherwise nothing (white) is plotted for these values.


    Uses pyplot.pcolormesh to do the plot. Any additional keyword arguments, in
    particular

      * cmap: defines the colormap

    are just passed on to this function.
    '''

    #Extract the map to plot, take the log if called for
    cmap = np.log10(pmap['map']) if log else pmap['map']

    #Mask invalid values
    cmap = np.ma.masked_invalid(cmap) if not invalid else cmap

    #Get the vertical range
    if not log and vmax is None:
        vmax = np.max(np.abs(np.array(cmap)[np.isfinite(cmap)]))
    if not log and vmin is None:
        vmin = -vmax if (cmap.min() < 0) else 0.

    #Get the energy range
    if emin is None:
        emin = pmap['ebins'][0]
    if emax is None:
        emax = pmap['ebins'][-1]

    #... and for zenith range
    if czmin is None:
        czmin = pmap['czbins'][0]
    if emax is None:
        czmax = pmap['czbins'][-1]

    #Use pcolormesh to be able to show nonlinear spaces
    x,y = np.meshgrid(pmap['czbins'],pmap['ebins'])
    plt.pcolormesh(x,y,cmap,vmin=vmin, vmax=vmax, **kwargs)

    #Add nice labels
    #if xlabel == None:
    #    plt.xlabel(r'cos(zenith)',fontsize=16)
    #else:
    plt.xlabel(xlabel,fontsize=fontsize)
    #if yabel == None:
    #    plt.ylabel('Energy [GeV]',fontsize=16)
    #else:
    plt.ylabel(ylabel,fontsize=fontsize)

    #And a title
    if title is not None:
        plt.suptitle(title,fontsize=fontsize)

    axis = plt.gca()
    #Check wether energy axis is linear or log-scale
    if logE is None:
        logE = is_logarithmic(pmap['ebins'])

    if logE:
        axis.semilogy()
    else:
        if not is_linear(pmap['ebins']):
           raise NotImplementedError('Bin edges appear to be neither logarithmically '
                         'nor linearly distributed!')


    #Make sure that the visible range does not extend beyond the provided range
    axis.set_ylim(emin,emax)
    axis.set_xlim(czmin,czmax)

    #Show the colorbar
    if cbar:
        col_bar = plt.colorbar(format=r'$10^{%.1f}$') if log else plt.colorbar()
        if zlabel:
            col_bar.set_label(zlabel,fontsize=fontsize)
        col_bar.ax.tick_params(labelsize=zlabel_size)
    #Return axes for further modifications
    return axis


def delta_map(amap, bmap):
    '''
    Calculate the differerence between the two maps (amap - bmap), and return as
    a map dictionary.
    '''
    if not np.allclose(amap['ebins'],bmap['ebins']) or \
       not np.allclose(amap['czbins'],bmap['czbins']):
       raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map'] - bmap['map'] }

def sum_map(amap, bmap):
    '''
    Calculate the sum of two maps (amap + bmap), and return as
    a map dictionary.
    '''
    if not np.allclose(amap['ebins'],bmap['ebins']) or \
       not np.allclose(amap['czbins'],bmap['czbins']):
       raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map'] + bmap['map'] }

def ratio_map(amap,bmap):
    '''
    Get the ratio of two maps (amap/bmap) and return as a map
    dictionary.
    '''
    if (not np.allclose(amap['ebins'],bmap['ebins']) or
        not np.allclose(amap['czbins'],bmap['czbins'])):
        raise ValueError('Map range does not match!')

    return { 'ebins': amap['ebins'],
             'czbins': amap['czbins'],
             'map' : amap['map']/bmap['map']}

def distinguishability_map(amap,bmap):
    '''
    Calculate the Akhmedov-Style distinguishability map of two maps,
    defined as amap-bmap/sqrt(amap)
    '''
    sqrt_map = {'ebins': amap['ebins'],
                'czbins': amap['czbins'],
                'map':np.sqrt(amap['map'])}
    return ratio_map(delta_map(amap,bmap),sqrt_map)


class PrintPlotFisher:
    '''
    A wrapper class around a Fisher matrix that allows to draw
    and pretty print in ipython
    '''

    def __init__(self, fisher, parnames=None, parvalues=None):
        '''Constructor takes
           - fisher: a Fisher matrix object
           - parnames: a list of display names for the parameters
           - parvalues: a list of parameter values (fiducial model)
        '''

        self.fisher = fisher
	#If both parameter names and fiducial model specified, take these
	if parnames != None and parvalues != None:
            self.parnames = parnames
            self.parvalues = parvalues
	#If not, take them from the Fisher matrix fisher
        else:
            self.parnames = self.fisher.parameters
            self.parvalues = self.fisher.best_fits

        #Some consistency checks
        if not (str(self.fisher.__class__) == 'pisa.analysis.fisher.Fisher.FisherMatrix'):
            raise ValueError('Expected FisherMatrix object, got %s instead'%(fisher.__class__))

	if parnames!=None:
            if not len(self.fisher.parameters)==len(self.parnames):
                raise IndexError('Number of parameters names does not match number of parameters! [%i, %i]' \
                    %(len(self.fisher.parameters), len(self.parnames)) )

	if parvalues!=None:
            if not len(self.fisher.parameters)==len(self.parvalues):
                raise IndexError('Number of default values does not match number of parameters! [%i, %i]' \
                    %(len(self.fisher.parameters), len(self.parvalues)) )


    def ipynb_pretty_print(self):
        '''
        Pretty print the matrix for the ipython notebook
        '''
        from IPython.display import Latex

        #Show the fiducial model
        outstr = r'Fiducial model: \begin{align}'
        for parname, parvalue in zip(self.parnames,self.parvalues):
            outstr += r' %s = %.2e \newline'%(parname,parvalue)
        outstr += r' \end{align} '

        #Now add the fisher matrix itself
        outstr += r'Fisher Matrix: $$ \mathcal{F} = \begin{vmatrix} '

        for row in self.fisher.matrix:
            for val in (row.flat[0:-1]).flat:
                  outstr += r' %.2e & '%val

            #Add last value with newline
            outstr += r' %.2e \newline'%row.flat[-1]

        outstr += r'\end{vmatrix} $$'

        return Latex(outstr)


    def draw(self, confLevels = [0.6827, 0.9545, 0.9973], parameters=None, fontsize=16, alphaCL = .75):
        ''' Uses matplotlib to create a triangular plot of all the 2d (marginalised) error ellipses
            for the specified joint confidence levels.
            - confLevels: The requested 2D confidence levels.
	    - parameters: List of parameter names (at least 3) for which the ellipses are
			  to be plotted (these need to exist in the underlying fisher matrix
			  object). If 'None', all parameters will be included.
	    - fontsize:	  Font size of plot title, labels, legend.
	    - alphaCL:    This allows controlling the transparencies of the overlaid ellipses,
		          alpha = 1 - alphaCL*confLevel. The default value should lead to reasonable
		          contrasts for typical confidence levels of 1, 2, 3 sigma.
        '''

	from matplotlib.patches import Ellipse, Rectangle
	from matplotlib.lines import Line2D
	from scipy.special import erfcinv
        # If no parameters are specified, just use all of them
        if parameters is None:
            parameters = self.fisher.parameters

        else:
            parameters = parameters

        # Collect parvalues and parlabels
        parvalues=[]
        parnames=[]
        for par in parameters:
            idx =  self.fisher.getParameterIndex(par)
            parvalues.append(self.parvalues[idx])
            parnames.append(self.parnames[idx])

        # Make a figure with size matched to the number of parameters
        nPar = len(parameters)
        size = min(nPar-1,8)*4
        fig = plt.figure(figsize=(size,size))
        # Remove space inbetween the subplots
        fig.subplotpars.wspace=0.
        fig.subplotpars.hspace=0.
        # Define the color arguments for the Ellipses
        ellipseArgs = { 'facecolor' : 'b',
                         'linewidth' : 0 }
        markerArgs = { 'marker':'o',
                       'markerfacecolor': 'r',
                       'linewidth': 0 }
        lineArgs = {'linestyle' : '--',
                    'color' : 'r' }

        # Loop over all parameters
        for idx1, par1 in enumerate(parameters):
            # Loop over all other parameters
            for idx2, par2 in list(enumerate(parameters))[idx1+1:]:

                # Make a new subplot in that subfigure
                axes = plt.subplot(nPar-1,nPar-1,idx2*(nPar-1) + (idx1-(nPar-2)))
                # Only show tick marks in the left-most column and bottom row
                axes.label_outer()
                axes.tick_params(which='both', labelsize=fontsize-2)

		err_ells = []
                # Draw the error ellipses for the requested confidence levels
		for CL in sorted(confLevels):
		    semiA, semiB, tan_2_th = self.fisher.getErrorEllipse(par1, par2, confLevel=CL)
		    tilt = np.rad2deg(np.arctan(tan_2_th)/2.)
		    err_ell = Ellipse(xy=(parvalues[idx1], parvalues[idx2]),
                                  width=2*semiA, height=2*semiB, angle=tilt,
                                  alpha=1.-alphaCL*CL, **ellipseArgs)
		    err_ells.append(err_ell)
                    axes.add_artist(err_ell)

                # Draw a red marker for the fiducial model
                plt.plot(parvalues[idx1],parvalues[idx2],**markerArgs)
                # Only set labels in the left-most column and bottom row
                if axes.is_last_row():
                    plt.xlabel(parnames[idx1],fontsize=fontsize)
                if axes.is_first_col():
                    plt.ylabel(parnames[idx2],fontsize=fontsize)

		# Restrict plot ranges to the lengths of the axes of the largest ellipse
		max_width = max([ell.width for ell in err_ells])
		max_height = max([ell.height for ell in err_ells])
		plt.xlim(parvalues[idx1]-1.1*max_width/2.,parvalues[idx1]+1.1*max_width/2.)
		plt.ylim(parvalues[idx2]-1.1*max_height/2.,parvalues[idx2]+1.1*max_height/2.)

		sigma1, sigma2 = self.fisher.getSigma(par1), self.fisher.getSigma(par2)
                # Show numbers for parameters in column on top
                if idx2 == idx1+1:
                    sigmaTot = sigma1
                    sigmaStat = self.fisher.getSigmaStatistical(par1)
                    sigmaSys = self.fisher.getSigmaSystematic(par1)
		    # For deltam31, 3 decimal digits are too little
		    if par1 != "deltam31":
			lab = "%s\n $\mathsf{= %.3f \pm %.3f(stat) \pm %.3f(sys)}$"\
                             %(parnames[idx1], parvalues[idx1], sigmaStat, sigmaSys)
		    else:
			lab = "%s\n $\mathsf{= [%.3f \pm %.3f(stat) \pm %.3f(sys)]\cdot 10^{-3}}$"\
			      %(parnames[idx1], parvalues[idx1]*(10**3), sigmaStat*(10**3), sigmaSys*(10**3))
		    plt.title(lab, fontsize=fontsize)

                # Now there is one parameter missing, that we show as right label
                # in the last row
                if axes.is_last_row() and axes.is_last_col():
                    sigmaTot = sigma2
                    sigmaStat = self.fisher.getSigmaStatistical(par2)
                    sigmaSys = np.sqrt(sigmaTot**2 - sigmaStat**2)
                    axes.yaxis.set_label_position('right')
		    # For deltam31, 3 decimal digits are too little
		    if par2 != "deltam31":
			lab = "%s\n $\mathsf{= %.3f \pm %.3f(stat) \pm %.3f(sys)}$"\
                              %(parnames[idx2], parvalues[idx2], sigmaStat, sigmaSys)
		    else:
			lab = "%s\n $\mathsf{= [%.3f \pm %.3f(stat) \pm %.3f(sys)]\cdot 10^{-3}}$"\
			      %(parnames[idx2], parvalues[idx2]*(10**3), sigmaStat*(10**3), sigmaSys*(10**3))

		    plt.ylabel(lab, fontsize=fontsize, horizontalalignment='center',
                              rotation=-90.,labelpad=+40)

                # Plot vertical and horizontal range to indicate one-sigma
                # marginalized levels
                plt.axvline(parvalues[idx1]-sigma1,**lineArgs)
                plt.axvline(parvalues[idx1]+sigma1,**lineArgs)
                plt.axhline(parvalues[idx2]-sigma2,**lineArgs)
                plt.axhline(parvalues[idx2]+sigma2,**lineArgs)

        # Use the top right corner to draw a legend
	if nPar>=3:
            axes = plt.subplot(nPar-1,nPar-1,nPar-1)
            axes.axison = False
	"""
	else: # we only have 2 parameters
	    axes = plt.subplot(111)
	    axes.axison = False
	"""
        # Create dummies for the legend objects
        legendObjs=[Line2D([0],[0],**markerArgs),
                    Line2D([0],[0],**lineArgs)]
        legendLabels=[r'default value',
                      r'$1\sigma$ stat.+syst.']

	# Convert given confidence levels to numbers of (double-sided) std. dev.
	sigmaLevels = np.sqrt(2)*erfcinv(np.subtract(1.,confLevels))
	for i in xrange(0,len(confLevels)):
	    sigma = sorted(sigmaLevels)[i]
            legendObjs.append(Rectangle([0,0],0,0,alpha=1.-alphaCL*sorted(confLevels)[i],**ellipseArgs))
            legendLabels.append(r'$%.1f\sigma$ conf. region'%sigma)

        # Draw a legend with all of these
        plt.legend(legendObjs,legendLabels,
                   loc='center',
                   numpoints=1,
                   frameon=False,
                   prop={'size':fontsize})

        return fig
