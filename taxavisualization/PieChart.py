# #!/usr/bin/env python
# PieChart.py

from __future__ import division
from matplotlib import use
use('agg')
from warnings import warn, filterwarnings
from TaxPlot import TaxPlot

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

# Sets up warning activity to always
filterwarnings("always")


class PieChart(TaxPlot):
    """Doc String Here"""
    # Sets up pie chart specific properties
    plot_ccw = False
    start_angle = 90
    show_labels = False
    numeric_labels = False
    label_distance = 1.1
    __labels = None
    __defaults = {'show_error': None,
                  'fig_dims': (5, 3),
                  'axis_dims': (0.06, 0.1, 0.48, 0.8),
                  'save_properties': {},
                  'show_edge': False,
                  'colors': None,
                  'show_legend': True,
                  'legend_offset': (1.8, 0.75),
                  'legend_properties': {},
                  'show_axes': True,
                  'show_frame': True,
                  'xlim': [-1.1, 1.1],
                  'ylim': [-1.1, 1.1],
                  'xticks': None,
                  'yticks': None,
                  'xticklabels': None,
                  'yticklabels': None,
                  'show_title': False,
                  'title_text': None,
                  'title_properties': {},
                  'use_latex': False,
                  'latex_family': 'sans-serif',
                  'latex_font': ['Helvetica', 'Arial'],
                  'plot_ccw': False,
                  'start_angle': 90,
                  'axis_lims': [-1.1, 1.1],
                  'show_labels': False,
                  'numeric_labels': False,
                  'label_distance': 1.1}

    def __init__(self, data, groups, samples, filename=None, **kwargs):
        """Initializes a PieChart instance"""
        # Sets up the piechart with a legend.
        self.fig_dims = (5, 3)
        self.axis_dims = (0.06, 0.1, 0.48, 0.8)
        self.show_legend = True
        self.legend_offset = (1.8, 0.75)
        self.xlim = [-1.1, 1.1]
        self.ylim = [-1.1, 1.1]
        # Initializes the object
        TaxPlot.__init__(self, data, groups, samples, error=None,
                         filename=filename, **kwargs)

        self.check_piechart()

    def check_piechart(self):
        """Checks the PieChart parameters are sane"""
        # Preforms an initial check of the data
        self.checkbase()

        # Checks the sanity of the plot_ccw argument
        if not isinstance(self.plot_ccw, bool):
            raise TypeError('plot_ccw must be boolian.')

        # Checks the sanity of the start_angle argument
        if not isinstance(self.start_angle, (int, float)):
            raise TypeError('start_angle must be an angle between 0 and '
                            '365 degrees.')
        if not 0 <= self.start_angle < 360:
            raise ValueError('start_angle must be an angle between 0 and '
                             '365 degrees.')

        # Checks the sanity of the show_labels argument
        if not isinstance(self.show_labels, bool):
            raise TypeError('show_labels must be boolian.')

        if not isinstance(self.numeric_labels, bool):
            raise TypeError('numeric_labels must be boolian.')

        # Checks the label distance
        if not isinstance(self.label_distance, (int, float)):
            raise TypeError('label distance must be numeric')

    def render_piechart(self):
        """Plots the data as a Pie Chart"""
        # Checks the object for rendering
        self.check_piechart()
        # Updates set fields
        self.update_colormap()
        self.update_dimensions()

        # Updates the latex rendering
        self.render_latex()

        # Warns the user if the data is more than one dimension
        data_shape = self.data.shape
        data_len = len(data_shape)
        if data_len > 1 and not (data_shape[0] == 1 or data_shape[1] == 1):
            warn('Data is a two-dimensional array. \nOnly the first '
                 'data column will be plotted.', UserWarning)
            self.data = self.data[:, 0]

        # Handles the data labels
        if self.show_labels and self.numeric_labels:
            labels = map(str, self.data)
        elif self.show_labels:
            labels = map(str, self.groups)
        else:
            labels = ['']*len(self.groups)

        # Plots the data clockwise
        [pie_patches, pie_labels] = \
            self._TaxPlot__axes.pie(x=self.data,
                                    labels=labels,
                                    labeldistance=self.label_distance,
                                    shadow=False,
                                    startangle=self.start_angle)

        # Colors the data
        label_font = self._TaxPlot__font_set['label']
        for idx, patch in enumerate(pie_patches):
            patch.set_facecolor(self._TaxPlot__colormap[idx, :])
            patch.set_edgecolor(self._TaxPlot__edgecolor[idx, :])
            pie_labels[idx].set_font_properties(label_font)

        # Updates the holding object
        self._TaxPlot__patches = pie_patches
        self._PieChart__labels = pie_labels

        # Handles the clockwise vs counter-clockwise properties of the axis
        if self.plot_ccw:
            self._TaxPlot__axes.set_xlim(reversed(self.xlim))
        else:
            self._TaxPlot__axes.set_xlim(self.xlim)
        self._TaxPlot__axes.set_ylim(self.ylim)
        self._TaxPlot__axes.set_visible(self.show_axes)

        # Adds a figure legend and a title.
        self.render_legend()
        self.render_title()
