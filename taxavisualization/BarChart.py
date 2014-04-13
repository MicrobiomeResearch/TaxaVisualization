# #!/usr/bin/env python
# BarChart.py
from __future__ import division
from numpy import arange, array
from matplotlib import use
use('agg')
from TaxPlot import TaxPlot

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"


class BarChart(TaxPlot):
    """Doc string here"""
    # Sets up barchart specific properties
    bar_width = 0.8
    xtick_interval = 1.0
    ytick_interval = 0.2
    xmin = -0.5
    x_font_angle = 45
    x_font_align = 'right'
    show_x_labels = True
    show_y_labels = True
    __bar_left = None
    __all_faces = []
    __defaults = {'show_error': None,
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
                  'bar_wdith': 0.8,
                  'xtick_interval': 1.0,
                  'ytick_interval': 0.2,
                  'xmin': -0.5,
                  'x_font_angle': 45,
                  'x_font_align': 'right',
                  'show_x_labels': True,
                  'show_y_labels': True}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        """Initializes a BarChart instance"""
        # Sets up the default figure and axis dimensions assuming there is
        # no legend displayed.
        self.fig_dims = (6, 4)
        self.axis_dims = (0.125, 0.1875, 0.75, 0.83334)
        # Initialzes the object
        TaxPlot.__init__(self, data, groups, samples, error, filename,
                         **kwargs)

        self.check_barchart()

    def check_barchart(self):
        """Checks the additional properties of BarChart instance are sane"""
        # Checks the base object initializes sanely
        self.checkbase()

        # Checks the bar_width class
        if not isinstance(self.bar_width, (int, float)):
            raise TypeError('bar_width must be a number')
        # Checks the xtick_interval class
        if not isinstance(self.xtick_interval, (int, float)):
            raise TypeError('xtick_interval must be a number')
        # Checks the bar_width is not greater than the xtick_interval
        if self.bar_width > self.xtick_interval:
            raise ValueError('The bar_width cannot be greater than the '
                             'xtick_interval.')

        # Checks the xmin argument
        if not isinstance(self.xmin, (int, float)):
            raise TypeError('xmin must be a number')

        # Checks x_font_angle is handled sanely
        if not isinstance(self.x_font_angle, (int, float)):
            raise TypeError('x_font_angle must be a number')
        if not 0 <= self.x_font_angle < 360:
            raise ValueError('x_font_angle is in degrees.\n'
                             'Values must be between 0 and 360,')

        # Checks the x_font_align argument
        alignments = set(['left', 'right', 'center'])
        if self.x_font_align not in alignments:
            raise ValueError('%s is not a supported font alignment.'
                             % self.x_font_align)

        # Checks show_x_labels is sane
        if not isinstance(self.show_x_labels, bool):
            raise TypeError('show_x_labels must be a boolian')

        # Checks show_y_labels is sane
        if not isinstance(self.show_y_labels, bool):
            raise TypeError('show_y_labels must be a boolian')

    def render_barchart(self):
        """Plots the data as a stacked barchart"""
        # Checks the input data object is sane
        self.check_barchart()
        # Updates variable properties
        self.set_colormap()
        self.set_dimensions()

        # Sets things up to render using LaTex
        self.render_latex()

        # Sets up the x_axis properties
        num_samples = len(self.samples)
        self.xticks = arange(num_samples) * self.xtick_interval
        xmax = max(self.xticks)+self.xtick_interval/2
        self.xlim = [self.xmin, xmax]
        # Updates the left side for the barchart
        self._BarChart__bar_left = self.xticks - self.bar_width/2

        # Plots the data
        for count, category in enumerate(self.data):
            bottom_bar = sum(self.data[0:count, :], 0)
            count_rev = len(self.groups) - (1 + count)
            facecolor = self._TaxPlot__colormap[count_rev, :]
            edgecolor = self._TaxPlot__edgecolor[count_rev, :]
            if self.show_error and self.error is not None:
                error = self.error[count, :]
            else:
                error = None
            faces = self._TaxPlot__axes.bar(left=self._BarChart__bar_left,
                                            height=category,
                                            width=self.bar_width,
                                            bottom=bottom_bar,
                                            color=facecolor,
                                            edgecolor=edgecolor,
                                            yerr=error,
                                            ecolor=array([0, 0, 0]))
            self._BarChart__all_faces.append(faces)
            self._TaxPlot__patches.append(faces[0])

        # Sets up the x tick labels
        if self.show_x_labels:
            self.xticklabels = self.samples
        else:
            self.xticklabels = ['']*num_samples

      # Gets the y axis properties if they are not known
        if self.ylim is None:
            self.ylim = self._TaxPlot__axes.get_ylim()
        if self.yticks is None:
            ymin = self.ylim[0]
            ymax = self.ylim[1]+self.ytick_interval
            self.yticks = arange(ymin, ymax, self.ytick_interval)
        if self.yticklabels is None and self.show_y_labels:
            self.yticklabels = map(str, self.yticks)
        elif self.yticklabels is None:
            self.yticklabels = ['']*len(self.yticks)

        # Updates the axis properties
        self._TaxPlot__axes.set_xlim(self.xlim)
        self._TaxPlot__axes.set_xticks(self.xticks)
        self._TaxPlot__axes.set_ylim(self.ylim)
        self._TaxPlot__axes.set_yticks(self.yticks)

        # Updates the label text
        tick_font = self._TaxPlot__font_set['tick']
        x_align = self.x_font_align
        self._TaxPlot__axes.set_xticklabels(self.xticklabels,
                                            fontproperties=tick_font,
                                            rotation=self.x_font_angle,
                                            horizontalalignment=x_align)
        self._TaxPlot__axes.set_yticklabels(self.yticklabels,
                                            fontproperties=tick_font)

        # Adds a legend
        self.render_legend()

        # Adds a title
        self.render_title()

        # Sets up xkcd
        self.render_xkcd()

