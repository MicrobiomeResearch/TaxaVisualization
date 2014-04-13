# #!/usr/bin/env python
# ScatterPlot.py

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


class ScatterPlot(TaxPlot):
    """Doc string here"""
    # Sets up properties for the distribution axes
    show_distribution = True
    show_dist_hist = False
    show_reg_line = False
    match_reg_line = False
    show_reg_equation = False
    show_error = False
    equation_position = ()
    show_r2 = False
    r2_position = ()
    connect_points = False
    x_axis_dims = (0.09375, 0.66667, 0.62500, 0.25000)
    y_axis_dims = (0.75000, 0.12500, 0.18750, 0.50000)
    markers = ['x', 'o', '.', '^', 's', '*']
    bins = 25
    round_to_x = 5
    round_to_y = 5
    __x_axes = None
    __y_axes = None
    __x_dist = None
    __x_bins = None
    __y_dist = []
    __y_bins = []
    __x_reg = None
    __y_reg = []
    __reg_stats = []
    __defaults = {'show_error': None,
                  'fig_dims': None,
                  'axis_dims': None,
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
                  'show_distribution': True,
                  'show_dist_hist': False,
                  'show_reg_line': False,
                  'match_reg_line': False,
                  'show_reg_equation': False,
                  'show_error': False,
                  'equation_position': (),
                  'show_r2': False,
                  'r2_position': (),
                  'connect_points': False,
                  'x_axis_dims': (0.09375, 0.66667, 0.62500, 0.25000),
                  'y_axis_dims': (0.75000, 0.12500, 0.18750, 0.50000),
                  'markers': ['x', 'o', '.', '^', 's', '*'],
                  'bins': 25,
                  'round_to_x': 5,
                  'round_to_y': 5}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        """Initializes a ScatterPlot instance"""
        # Sets up the axis dimensions and figure dimensions, since the defaults
        # are different for a scatter plot instance
        if 'fig_dims' not in kwargs:
            kwargs['fig_dims'] = (8, 6)
        if 'axis_dims' not in kwargs:
            kwargs['axis_dims'] = (0.09375, 0.125, 0.625, 0.666667)

        # Initializes an instance of a ScatterPlot object
        TaxPlot.__init__(self, data, groups, samples, error, filename,
                         **kwargs)
        # Updates the figure dimensions appropriately for the formatted axes.
        # Current axes are removed and updated with the new dimensions
        self._TaxPlot__fig.clf()
        self.update_scatter_dimensions()

    def check_scatter(self):
        """Checks additional properties of the scatter instance are sane"""
        # Checks show distribution
        self.checkbase()

        # Checks the show distribution argument
        if not isinstance(self.show_distribution, bool):
            raise TypeError('show_distribution must be a bool')

        # Checks the show_dist_hist argument
        if not isinstance(self.show_dist_hist, bool):
            raise TypeError('show_dist_hist must be a bool')

        # Checks the show_reg_line argument
        if not isinstance(self.show_reg_line, bool):
            raise TypeError('show_reg_line must be a bool')

        # Checks the match_reg_line argument
        if not isinstance(self.match_reg_line, bool):
            raise TypeError('match_reg_line must be a bool')

        # Checks the show_reg_equation argument
        if not isinstance(self.show_reg_equation, bool):
            raise TypeError('show_reg_equation must be a bool')

        # Checks the show_r2 argument
        if not isinstance(self.show_r2, bool):
            raise TypeError('show_r2 must be a bool')

        # Checks the show_error argument
        if not isinstance(self.show_error, bool):
            raise TypeError('show_error must be a bool')

        # Checks the connect_points argument
        if not isinstance(self.connect_points, bool):
            raise TypeError('connect_points must be a bool')

        # Checks the equation_position argument
        if not isinstance(self.equation_position, tuple):
            raise TypeError('equation_position must be a tuple')

        # Checks the r2_position argument
        if not isinstance(self.r2_position, tuple):
            raise TypeError('r2_position must be a tuple')

        # Checks the x_axis_dims argument
        if not isinstance(self.x_axis_dims, tuple):
            raise TypeError('x_axis_dims must be a tuple')

        # Checks the y_axis_dims argument
        if not isinstance(self.y_axis_dims, tuple):
            raise TypeError('y_axis_dims must be a tuple')

        # Checks the marker class
        if not isinstance(self.markers, (str, list, tuple)):
            raise TypeError('markers must be a string or interable class '
                            'of strings.')
        if isinstance(self.markers, (list, tuple)):
            for m in self.markers:
                if not isinstance(m, str):
                    raise ValueError('markers must be a string or interable'
                                     ' class of strings.')

        # Checks the bins argument
        if not isinstance(self.bins, int):
            raise TypeError('bins must be an integer')
        if self.bins < 1:
            raise ValueError('There must be at least one bin')

        # Checks the round_to argument
        if not isinstance(self.round_to_x, (int, float)):
            raise TypeError('round_to_x must be a number')
        if self.round_to_x < 0:
            raise ValueError('round_to_x must be positive')

        # Checks the round_to argument
        if not isinstance(self.round_to_y, (int, float)):
            raise TypeError('round_to_y must be a number')
        if self.round_to_y < 0:
            raise ValueError('round_to_y must be positive')

        # Checks the range arguments
        if self.xlim is not None:
            if not isinstance(self.xlim, (list, tuple)):
                raise TypeError('xlim must be a two-element tuple or list')
            if not len(self.xlim) == 2:
                raise ValueError('xlim must be a two-element tuple or list')
            x_min_num = isinstance(self.xlim[0], (int, float))
            x_max_num = isinstance(self.xlim[1], (int, float))
            if not (x_min_num and x_max_num):
                raise TypeError('The values in xlim must be numbers')
            if not self.xlim[0] < self.xlim[1]:
                raise ValueError('The x min must be the first value in '
                                 'xlim')

        if self.ylim is not None:
            if not isinstance(self.ylim, (list, tuple)):
                raise TypeError('ylim must be a two-element tuple or list')
            if not len(self.ylim) == 2:
                raise ValueError(' ylim must be a two-element tuple or'
                                 ' list')
            y_min_num = isinstance(self.ylim[0], (int, float))
            y_may_num = isinstance(self.ylim[1], (int, float))
            if not (y_min_num and y_may_num):
                raise TypeError('The values in ylim must be numbers')
            if not self.ylim[0] < self.ylim[1]:
                raise ValueError('The y min must be the first value in '
                                 'ylim')

    def update_scatter_dimensions(self):
        """Updates the axis and figure dimensions for a distribution set."""
        # Checks the object and properties
        self.check_scatter()

        # Updates the figure dimensions
        self._TaxPlot__fig.set_size_inches(self.fig_dims)
        # Adds the main axis to the figure
        ax_pos = self.axis_dims
        if isinstance(self._TaxPlot__axes, plt.Axes):
            self._TaxPlot__axes.set_position(ax_pos)
        else:
            self._TaxPlot__axes = self._TaxPlot__fig.add_axes(ax_pos)
        # Adds the x and y axes if appropriate
        if self.show_distribution:
            if isinstance(self._ScatterPlot__x_axes, plt.Axes):
                self._ScatterPlot__x_axes.set_position(self.x_axis_dims)
            else:
                self._ScatterPlot__x_axes = \
                    self._TaxPlot__fig.add_axes(self.x_axis_dims)

            if isinstance(self._ScatterPlot__y_axes, plt.Axes):
                self._ScatterPlot__y_axes.set_position(self.y_axis_dims)
            else:
                self._ScatterPlot__y_axes = \
                    self._TaxPlot__fig.add_axes(self.y_axis_dims)
        else:
            self._ScatterPlot__x_axes = None
            self._ScatterPlot__y_axes = None

        # Removes any axes which should not be present
        fig_axes = self._TaxPlot__fig.get_axes()
        for axis in fig_axes:
            # Determines if identified axis is an accepted axis
            axis_bounds = axis.get_position().bounds
            is_main = axis_bounds == self.axis_dims
            is_x_ax = axis_bounds == self.x_axis_dims
            is_y_ax = axis_bounds == self.y_axis_dims
            if not (is_main and is_x_ax and is_y_ax):
                ax = axis
                ax.set_axis_off()

    def calculate_smoothed_range(self):
        """Caclulates a corrected range for a dataset"""
        # Calculates the range for the distributions
        if self.xlim is None:
            x_min = floor(min(self.groups)/self.round_to_x)*self.round_to_x
            x_max = ceil(max(self.groups)/self.round_to_x)*self.round_to_x
            self.xlim = [x_min, x_max]
        if self.ylim is None:
            y_min = floor(self.data.min()/self.round_to_y)*self.round_to_y
            y_max = ceil(self.data.max()/self.round_to_y)*self.round_to_y
            self.ylim = [y_min, y_max]

    def calculate_distribution(self):
        """Calculates the x and y histogram from for the data"""
        # Makes sure the object is sane
        self.check_scatter()

        # Gets the corrected ranges
        self.calculate_smoothed_range()

        # Calculates the independent distribution
        (x_dist, x_bound) = histogram(self.groups,
                                      bins=self.bins,
                                      range=self.xlim)
        self._ScatterPlot__x_dist = x_dist/sum(x_dist)
        if self.show_dist_hist:
            self._ScatterPlot__x_bins = x_bound[:-1]
        else:
            xcenter = []
            for idx, x in enumerate(x_bound[:-1]):
                xcenter.append((x_bound[idx]+x_bound[idx+1])/2)
            self._ScatterPlot__x_bins = xcenter

        # Calculates the dependent distribution(s)
        if len(self.data.shape) == 1:
            (y_dist, y_bound) = histogram(self.data,
                                          bins=self.bins,
                                          range=self.ylim)
            self._ScatterPlot__y_dist = [y_dist/sum(y_dist)]
            if self.show_dist_hist:
                self._ScatterPlot__y_bins = [y_bound[:-1]]
            else:
                center = []
                for idx, y in enumerate(y_bound[:-1]):
                    center.append((y+y_bound[idx+1])/2)
                self._ScatterPlot__y_bins = [center]
        else:
            bins_list = []
            dist_list = []
            for idx, samp in enumerate(self.samples):
                (y_dist, y_bound) = histogram(self.data[:, idx],
                                              bins=self.bins,
                                              range=self.ylim)
                dist_list.append(y_dist/sum(y_dist))
                if self.show_dist_hist:
                    bins_list.append(y_bound[:-1])
                else:
                    center = []
                    for idx, y in enumerate(y_bound[:-1]):
                        center.append((y+y_bound[idx+1])/2)
                    bins_list.append(center)
            self._ScatterPlot__y_bins = bins_list
            self._ScatterPlot__y_dist = dist_list

    def calculate_regression(self):
        """Calculates a linear regression correlating the groups to the data"""
        # Checks the data is sane
        self.check_scatter()
        # Sets the limits if not already known
        self.calculate_smoothed_range()
        # Gets x values
        x_int = self.round_to_x*0.1
        self._ScatterPlot__x_reg = arange(self.xlim[0], self.xlim[1]+x_int,
                                          x_int)

        # Calculates the series of linear regressions for each of the data
        # columns
        if len(self.data.shape) == 1:
            reg_stats = linregress(self.groups, self.data)
            m = reg_stats[0]
            b = reg_stats[1]
            self._ScatterPlot__reg_stats = [reg_stats]
            self._ScatterPlot__y_reg = [m*self._ScatterPlot__x_reg+b]
        else:
            hold_stats = []
            hold_reg = []
            for idx, sample in enumerate(self.samples):
                reg_stats = linregress(self.groups, self.data[:, idx])
                hold_stats.append(reg_stats)
                m = reg_stats[0]
                b = reg_stats[1]
                hold_reg.append(m*self._ScatterPlot__x_reg+b)
            self._ScatterPlot__reg_stats = hold_stats
            self._ScatterPlot__y_reg = hold_reg

    def render_scatterplot(self):
        """Plots data as a scatter plot or trace"""
        # Checks the data
        self.check_scatter()
        # Updates figure properties
        self.update_scatter_dimensions()
        self.update_colormap()
        # Calculates range properties
        self.calculate_smoothed_range()
        self.calculate_distribution()
        self.calculate_distribution()

        # Sets up LaTeX rendering
        self.render_latex()

        print self.data
        print self.groups
        print len(self.data) == len(self.groups)

        # plt.plot([1, 2, 3], [1, 2, 3])

    #     # Sets up some plotting characteristics
    #     if self.connect_points:
    #         linestyle = '-'
    #     else:
    #         linestyle = None

    #     # Plots vector data
    #     self._TaxPlot__axes.set_visible(True)
    # # if len(self.data.shape) == 1:
    #     color = self._TaxPlot__colormap[0, :]
    #     self._TaxPlot__patches = \
    #         self._TaxPlot__axes.plot(x=self.groups, y=self.data)
        # if self.show_error and self.error is not None:
        #     self._TaxPlot__axes.errorbar(x=self.groups, y=self.data,
        #                                  yerr=self.error,
        #                                  ecolor=color, fmt=None)
        # if self.show_reg_line:
        #     self._TaxPlot__axes.plot(x=self._ScatterPlot__x_reg,
        #                              y=self._ScatterPlot__y_reg[0])
