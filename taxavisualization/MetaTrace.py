from __future__ import division
from numpy import array, histogram, floor, ceil, arange
from scipy.stats import linregress
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
from TaxPlot import TaxPlot

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"


class MetaTrace(TaxPlot):
    """Doc String Here"""
    # Sets up properties for the distribtuion
    linestyle = [None]
    markers = ['x', 'o', '.', '^', '*']
    __defaults = {'show_error': True,
                  'fig_dims': (6, 4),
                  'axis_dims': (0.125, 0.1875, 0.75, 0.83334),
                  'save_properties': {},
                  'show_edge': False,
                  'colors': None,
                  'show_legend': False,
                  'legend_offset': None,
                  'legend_properties': {},
                  'show_axes': True,
                  'show_frame': True,
                  'axis_properties': {},
                  'show_title': False,
                  'title_text': None,
                  'title_properties': {},
                  'use_latex': False,
                  'latex_family': 'sans-serif',
                  'latex_font': ['Helvetica', 'Arial'],
                  'linestyle': None,
                  'markers': ['x', 'o', '.', '^', 's', '*']}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        # Sets up the default values
        if 'fig_dims' not in kwargs:
            kwargs['fig_dims'] = (6, 4)
        if 'axis_dims' not in kwargs:
            kwargs['axis_dims'] = (0.125, 0.1875, 0.75, 0.83334)
        if 'show_error' not in kwargs:
            kwargs['show_error'] = True
        # Intitialzes an instance of the object
        TaxPlot.__init__(self, data, groups, samples, error, filename,
                         **kwargs)

    def check_trace(self):
        """Checks the MetaTrace object initalizes sanely"""
        # Checks the base object
        self.checkbase()

        # Checks the linestyle argument
        if not isinstance(self.linestyle, list):
            raise TypeError('linestyle must be a list.')
        if len(self.linestyle) == 0:
            raise ValueError('linestyle cannot be empty')
        for ls in self.linestyle:
            if ls is not None and ls not in set(['-', '--', '-.', ':']):
                raise TypeError('linestyles is not supported')
        if 1 < len(self.linestyle) <= len(self.data.shape):
            raise ValueError('There is not a linestyle for each data set.')
        elif len(self.linestyle) == 1:
            print len(self.samples)
            self.linestyle = self.linestyle*len(self.samples)
            print self.linestyle

        # Checks the marker argument
        if not isinstance(self.markers, list):
            raise TypeError('markers must be a list.')
        if len(self.markers) == 0:
            raise ValueError('markers cannot be empty')
        for m in self.markers:
            marker_set = set(['.', ',', '^', 'v', '>', '<', '1', '2', '3', 's',
                              '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd',
                              '|', '_', 'o'])
            if m is not None and m not in marker_set:
                raise TypeError('%s is not a supported marker.' % m)
        if 1 < len(self.markers) <= len(self.data.shape):
            raise ValueError('There is not a marker for each data set.')
        elif len(self.markers) == 1:
            self.markers = self.markers*len(self.samples)

    # def render_trace(self):
    #     """Creates the trace figure"""
    #     print self.markers
    #     print self.linestyle
    #     # Checks the input data object is sane
    #     self.check_trace()
    #     # Updates variable properties
    #     self.update_colormap()
    #     self.update_dimensions()

    #     self._TaxPlot__axes.set_visible(True)

    #     # Sets things up to render using LaTex
    #     self.render_latex()
    #     # Plots the data
    #     for count, category in enumerate(self.samples):
    #         color = self._TaxPlot__colormap[:, count]
    #         lines = self._TaxPlot__axes.plot(x=self.groups,
    #                                          y=self.data[:, count],
    #                                          marker=self.markers[count],
    #                                          linestyle=self.linestyle[count],
    #                                          color=color,
    #                                          label=self.samples[count])
    #         self._TaxPlot__patches.append(lines)
        
        
        