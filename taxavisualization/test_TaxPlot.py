# #!/usr/bin/env python
# test_TaxPlot.py

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

from unittest import TestCase, main
from warnings import (filterwarnings, catch_warnings)
from os import remove
from os.path import realpath, dirname, join as pjoin, exists
from numpy import array, zeros, arange, log10
from numpy.testing import assert_array_equal, assert_almost_equal
from matplotlib import use
use('agg', warn=False)
from matplotlib.legend import Legend
from matplotlib.text import Text
from matplotlib.patches import Rectangle, Wedge
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from TaxPlot import TaxPlot, BarChart, PieChart
from MetaTrace import MetaTrace
from ScatterPlot import ScatterPlot
from americangut.make_phyla_plots import translate_colorbrewer

# Determines the location fo the reference files
TEST_DIR = dirname(realpath(__file__))

# Sets up warning activity to always
filterwarnings("always")


class TestTaxPlot(TestCase):

    def setUp(self):
        """Sets up variables for testing"""
        # Sets up the group distribution
        self.data = array([[0.1, 0.2, 0.3],
                           [0.2, 0.3, 0.4],
                           [0.3, 0.4, 0.1],
                           [0.4, 0.1, 0.2]])
        self.samples = ['Harry', 'Ron', 'Hermione']
        self.groups = ['Snape', 'D_Malfoy', 'Umbridge', 'Voldemort']
        self.error = array([[0.100, 0.100, 0.100],
                            [0.050, 0.050, 0.050],
                            [0.010, 0.010, 0.010],
                            [0.005, 0.005, 0.005]])
        # Sets up continous distribution
        self.ind = array([01.000,  2.000,  3.000,  4.000,  5.000,
                          06.000,  7.000,  8.000,  9.000, 10.000,
                          11.000, 12.000, 13.000, 14.000, 15.000,
                          16.000, 17.000, 18.000, 19.000, 20.000,
                          21.000, 22.000, 23.000, 24.000, 25.000])
        self.dep = array([[5.667,  5.960,  6.501,  7.273,   7.346,
                           08.075,  8.773,  8.799,  9.543, 09.904,
                           10.634, 10.946, 11.545, 12.114, 12.542,
                           12.927, 13.579, 13.935, 14.486, 14.945,
                           15.300, 15.895, 16.623, 16.940, 17.459],
                          [11.013, 10.956,  8.248, 10.850, 10.619,
                           07.635,  7.258,  8.113,  6.022,  7.431,
                           06.274,  6.865,  5.389,  4.705,  5.352,
                           02.880,  3.715,  1.452,  1.426,  1.028,
                           01.465, -0.644,  0.998,  0.985,  -0.587]
                          ]).transpose()
        self.err = array([[0.320,  0.465,  0.674,  0.445,  0.433,
                           0.498,  0.552,  0.344,  0.758,  0.542,
                           0.594,  0.378,  0.390,  0.441,  0.378,
                           0.503,  0.446,  0.445,  0.545,  0.537,
                           0.456,  0.361,  0.562,  0.486,  0.506],
                          [0.018,  0.011,  0.239,  0.662,  0.552,
                           0.790,  1.115,  0.055,  0.211,  0.201,
                           0.257,  0.190,  0.052,  0.148,  0.118,
                           0.312,  0.017,  0.086,  0.470,  0.158,
                           0.025,  0.458,  0.368,  1.009,  0.017]
                          ]).transpose()
        # Sets up figure properties
        self.fig_dims = (3, 4)
        self.axis_dims = (0.3, 0.3, 0.6, 0.6)
        self.colormap = array([[0.00, 0.00, 0.00],
                               [0.25, 0.25, 0.25],
                               [0.50, 0.50, 0.50],
                               [0.75, 0.75, 0.75]])
        self.params_file = "TaxPlot:colors\t'Spectral'\nTaxPlot:fig_dims\t"\
                           "(6, 4)"
        # Generates test instances
        self.base_test = TaxPlot(fig_dims=self.fig_dims,
                                 axis_dims=self.axis_dims,
                                 data=self.data,
                                 groups=self.groups,
                                 samples=self.samples,
                                 error=self.error)

        self.bar_test = BarChart(fig_dims=self.fig_dims,
                                 axis_dims=self.axis_dims,
                                 data=self.data,
                                 groups=self.groups,
                                 samples=self.samples,
                                 error=self.error)

        self.pie_test = PieChart(data=self.data[:, 0],
                                 groups=self.groups,
                                 samples=[self.samples[0]])

        self.trace_test = MetaTrace(data=self.dep,
                                    groups=self.ind,
                                    error=self.err,
                                    samples=['Gryffendors', 'Slythrins'],
                                    colors=array([[1, 0.0, 0],
                                                  [0, 0.5, 0]]))

        self.scatter_test = ScatterPlot(data=self.dep,
                                        groups=self.ind,
                                        error=self.err,
                                        samples=['Venom Activity',
                                                 'Antivsenom activity'])
        # Sets up known test properties
        self.base_properties = set(['data', 'groups', 'samples', 'error',
                                    'show_error', '_TaxPlot__error_bars',
                                    'fig_dims',
                                    '_TaxPlot__fig', 'axis_dims',
                                    '_TaxPlot__axes', '_TaxPlot__filepath',
                                    '_TaxPlot__filename', '_TaxPlot__filetype',
                                    'save_properties', 'show_edge',
                                    'colors', '_TaxPlot__colormap',
                                    '_TaxPlot__edgecolor',
                                    'show_legend', 'legend_offset',
                                    'legend_properties',
                                    '_TaxPlot__patches',
                                    '_TaxPlot__legend', 'show_axes',
                                    'show_frame', 'xlim', 'ylim', 'xticks',
                                    'yticks', 'xticklabels', 'yticklabels',
                                    'xlabel', 'ylabel',
                                    'show_title', 'title_text',
                                    'title_properties', '_TaxPlot__title',
                                    'use_latex', 'latex_family', 'latex_font',
                                    '_TaxPlot__font_set',
                                    'priority', '_TaxPlot__properties',
                                    '_TaxPlot__defaults'])
        self.bar_properties = set(['bar_width', 'xmin', 'xtick_interval',
                                   'ytick_interval', 'x_font_angle',
                                   'x_font_align', 'show_x_labels',
                                   'show_y_labels', '_BarChart__bar_left',
                                   '_BarChart__all_faces',
                                   '_BarChart__defaults'
                                   ]).union(self.base_properties)
        self.pie_properties = set(['plot_ccw', 'start_angle',
                                   'show_labels', 'numeric_labels',
                                   'label_distance', '_PieChart__labels',
                                   '_PieChart__defaults'
                                   ]).union(self.base_properties)
        self.trace_properties = set(['linestyle', 'markers'
                                     ]).union(self.base_properties)
        self.scatter_properties = set(['show_distribution',
                                       'show_dist_hist',
                                       'show_reg_line', 'match_reg_line',
                                       'show_reg_equation', 'show_error',
                                       'show_r2', 'connect_points',
                                       'equation_position', 'r2_position',
                                       'connect_points', 'x_axis_dims',
                                       'y_axis_dims', 'markers', 'bins',
                                       'round_to_x', 'round_to_y',
                                       '_ScatterPlot__x_axes',
                                       '_ScatterPlot__y_axes',
                                       '_ScatterPlot__x_dist',
                                       '_ScatterPlot__y_dist',
                                       '_ScatterPlot__x_bins',
                                       '_ScatterPlot__y_bins',
                                       '_ScatterPlot__x_reg',
                                       '_ScatterPlot__y_reg',
                                       '_ScatterPlot__reg_stats', 'priority',
                                       '_ScatterPlot__defaults',
                                       '_TaxPlot__defaults',
                                       ]).union(self.base_properties)

    def tearDown(self):
        """Handles teardown of the test object"""
        plt.close('all')

    # # Test intilization of classes
    # def test_base_init(self):
    #     """Tests the base object initializes correctly"""
    #     # Sets up known values for default properties
    #     show_error = False
    #     show_edge = False
    #     colors = None
    #     show_legend = False
    #     legend_offset = None
    #     legend_properties = {}
    #     show_axes = True
    #     show_frame = True
    #     xlim = None
    #     ylim = None
    #     xticks = None
    #     yticks = None
    #     xticklabels = None
    #     yticklabels = None
    #     show_title = False
    #     title_text = None
    #     title_properties = {}
    #     use_latex = False
    #     latex_family = 'sans-serif'
    #     latex_font = ['Helvetica', 'Arial']
    #     # Compares the properties to the known
    #     self.assertTrue((self.base_test.data == self.data).all())
    #     self.assertEqual(self.base_test.groups, self.groups)
    #     self.assertEqual(self.base_test.samples, self.samples)
    #     self.assertTrue((self.base_test.error == self.error).all())
    #     self.assertEqual(self.base_test.show_error, show_error)
    #     self.assertEqual(self.base_test.colors, colors)
    #     self.assertEqual(self.base_test.show_edge, show_edge)
    #     self.assertEqual(self.base_test.show_legend, show_legend)
    #     self.assertEqual(self.base_test.legend_offset, legend_offset)
    #     self.assertEqual(self.base_test.legend_properties, legend_properties)
    #     self.assertEqual(self.base_test.show_axes, show_axes)
    #     self.assertEqual(self.base_test.show_frame, show_frame)
    #     self.assertEqual(self.base_test.xlim, xlim)
    #     self.assertEqual(self.base_test.ylim, ylim)
    #     self.assertEqual(self.base_test.xticks, xticks)
    #     self.assertEqual(self.base_test.xticklabels, xticklabels)
    #     self.assertEqual(self.base_test.yticks, yticks)
    #     self.assertEqual(self.base_test.yticklabels, yticklabels)
    #     self.assertEqual(self.base_test.show_title, show_title)
    #     self.assertEqual(self.base_test.title_text, title_text)
    #     self.assertEqual(self.base_test.title_properties, title_properties)
    #     self.assertEqual(self.base_test.use_latex, use_latex)
    #     self.assertEqual(self.base_test.latex_family, latex_family)
    #     self.assertEqual(self.base_test.latex_font, latex_font)
    #     self.assertEqual(self.base_properties,
    #                      self.base_test._TaxPlot__properties)

    # def test_bar_init(self):
    #     """Test BarChart objects are initialized correctly"""
    #      # Sets up known values for default properties
    #     bar_width = 0.8
    #     xmin = -0.5
    #     xtick_interval = 1.0
    #     ytick_interval = 0.2
    #     x_font_angle = 45
    #     x_font_align = 'right'
    #     # Compares the properties to the known
    #     self.assertEqual(self.bar_test.bar_width, bar_width)
    #     self.assertEqual(self.bar_test.xmin, xmin)
    #     self.assertEqual(self.bar_test.xtick_interval, xtick_interval)
    #     self.assertEqual(self.bar_test.ytick_interval, ytick_interval)
    #     self.assertEqual(self.bar_test.x_font_angle, x_font_angle)
    #     self.assertEqual(self.bar_test.x_font_align, x_font_align)
    #     self.assertEqual(self.bar_properties,
    #                      self.bar_test._TaxPlot__properties)

    # def test_pie_init(self):
    #     """Checks PieChart Object initialize correctly"""
    #     # Sets up known default values
    #     plot_ccw = False
    #     start_angle = 90
    #     show_labels = False
    #     numeric_labels = False
    #     label_distance = 1.1
    #     # Compares the properties to the known
    #     self.assertEqual(self.pie_test.plot_ccw, plot_ccw)
    #     self.assertEqual(self.pie_test.start_angle, start_angle)
    #     self.assertEqual(self.pie_test.show_labels, show_labels)
    #     self.assertEqual(self.pie_test.numeric_labels, numeric_labels)
    #     self.assertEqual(self.pie_test.label_distance, label_distance)
    #     self.assertEqual(self.pie_properties,
    #                      self.pie_test._TaxPlot__properties)

    # def test_trace_init(self):
    #     """Checks a MetaTrace object initialzies correctly"""
    #     linestyle = [None]
    #     markers = ['x', 'o', '.', '^', '*']
    #     self.assertEqual(self.trace_test.linestyle, linestyle)
    #     self.assertEqual(self.trace_test.markers, markers)

    # def test_scatter_init(self):
    #     """Checks a ScatterChart object initializes sanely"""
    #     # Sets up the known default values
    #     show_distribution = True
    #     show_dist_hist = False
    #     show_reg_line = False
    #     match_reg_line = False
    #     show_reg_equation = False
    #     show_error = False
    #     equation_position = ()
    #     show_r2 = False
    #     r2_position = ()
    #     connect_points = False
    #     x_axis_dims = (0.09375, 0.66667, 0.62500, 0.25000)
    #     y_axis_dims = (0.75000, 0.12500, 0.18750, 0.50000)
    #     markers = ['x', 'o', '.', '^', 's', '*']
    #     bins = 25
    #     round_to_x = 5
    #     round_to_y = 5
    #     # Compares the known default properties to the instance properties
    #     self.assertEqual(self.scatter_test.show_distribution,
    #                      show_distribution)
    #     self.assertEqual(self.scatter_test.show_dist_hist,
    #                      show_dist_hist)
    #     self.assertEqual(self.scatter_test.show_reg_line, show_reg_line)
    #     self.assertEqual(self.scatter_test.match_reg_line, match_reg_line)
    #     self.assertEqual(self.scatter_test.show_reg_equation,
    #                      show_reg_equation)
    #     self.assertEqual(self.scatter_test.show_error, show_error)
    #     self.assertEqual(self.scatter_test.show_r2, show_r2)
    #     self.assertEqual(self.scatter_test.connect_points, connect_points)
    #     self.assertEqual(self.scatter_test.equation_position,
    #                      equation_position)
    #     self.assertEqual(self.scatter_test.r2_position, r2_position)
    #     self.assertEqual(self.scatter_test.x_axis_dims,
    #                      x_axis_dims)
    #     self.assertEqual(self.scatter_test.y_axis_dims,
    #                      y_axis_dims)
    #     self.assertEqual(self.scatter_test.markers, markers)
    #     self.assertEqual(self.scatter_test.bins, bins)
    #     self.assertEqual(self.scatter_test.round_to_x, round_to_x)
    #     self.assertEqual(self.scatter_test.round_to_y, round_to_y)
    #     self.assertEqual(self.scatter_properties,
    #                      self.scatter_test._TaxPlot__properties)

    # def test_add_attributes(self):
    #     """Checks that keyword argument attributes can be added correctly"""
    #     show_legend = True
    #     self.base_test.add_attributes(show_legend=show_legend)
    #     self.assertEqual(self.base_test.show_legend, show_legend)

    # # Tests checkbase
    # def test_checkbase_show_error_class(self):
    #     """Tests checkbase throws an error when show_error is not a bool"""
    #     self.base_test.show_error = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_save_properties_class(self):
    #     """"Checks the save_properties class handling is sane"""
    #     self.base_test.save_properties = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_fig_classs(self):
    #     """Checks that an error is called when fig_dims class is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = 'foo'
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_fig_dim_classs(self):
    #     """Checks that an error is called when fig_dims class is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = ('foo')
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_num_fig_dims(self):
    #     """Checks that an error is called when fig_dims length is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = (1, 2, 3)
    #     # Checks an error is thrown
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_axis_class(self):
    #     """Checks an error is called when axis_dims is of the wrong class"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.axis_dims = 'foo'
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_axis_dims_error(self):
    #     """Checks an error is called when axis_dims is of the wrong class"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.axis_dims = (1, 2, 3)
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_show_edge_class(self):
    #     """Checks the show_edge class checking is sane"""
    #     self.base_test.show_edge = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_show_legend_class(self):
    #     """Checks the show_edge class checking is sane"""
    #     self.base_test.show_legend = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_legend_properties(self):
    #     """Checks sanity of legend_properties class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.legend_properties = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_legend_offset(self):
    #     """Checks sanity of legend_offset checking"""
    #     # Checks the show_title handling is sane
    #     self.base_test.legend_offset = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)
    #     self.base_test.legend_offset = ('Foo', 'Bar', 'Cat')
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_show_axes(self):
    #     """Checks sanity of show_axes class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_axes = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_show_frame(self):
    #     """Checks sanity of show_frame class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_frame = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xlim_class(self):
    #     """Tests checkbase errors when xlim is not a list or None value"""
    #     self.base_test.xlim = 'Harry'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xlim_length(self):
    #     """Tests checkbase errors when xlim does not have two elements"""
    #     self.base_test.xlim = [0]
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_xlim_numeric(self):
    #     """Tests checkbase errors when xlim is not a list of numerics"""
    #     self.base_test.xlim = [0, 'foo']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_ylim_class(self):
    #     """Tests checkbase errors when xlim is not a list or None value"""
    #     self.base_test.ylim = 'Harry'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_ylim_length(self):
    #     """Tests checkbase errors when xlim does not have two elements"""
    #     self.base_test.ylim = [0]
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_ylim_numeric(self):
    #     """Tests checkbase errors when xlim is not a list of numerics"""
    #     self.base_test.ylim = [0, 'foo']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xtick_class(self):
    #     """Tests checkbase error swhen xtick is not a list"""
    #     self.base_test.xticks = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xtick_list_of_numbers(self):
    #     """Tests checkbase error swhen xtick is not a list of numbers"""
    #     self.base_test.xticks = ['foo', 'bar']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xticklabels_class(self):
    #     """Tests checkbase errors when xticklabels is not a list"""
    #     self.base_test.xticklabels = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_xticklabels_list_of_strings(self):
    #     """Tests checkbase errors when xticklabels is not a list of strings"""
    #     self.base_test.xticklabels = [1, 2, 3]
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_no_xticks_and_xticklabels(self):
    #     """Tests checkbase errors when xticklabels is sane but no xticks"""
    #     self.base_test.xticks = None
    #     self.base_test.xticklabels = ['foo', 'bar']
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_diff_length_xticks_and_xticklabels(self):
    #     """Tets checkbase errors when xtick and xticklabels are diff length"""
    #     self.base_test.xticks = [1, 2, 3]
    #     self.base_test.xticklabels = ['foo', 'bar']
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_ytick_class(self):
    #     """Tests checkbase error swhen ytick is not a list"""
    #     self.base_test.yticks = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_ytick_list_of_numbers(self):
    #     """Tests checkbase error swhen ytick is not a list of numbers"""
    #     self.base_test.yticks = ['foo', 'bar']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_yticklabels_class(self):
    #     """Tests checkbase errors when yticklabels is not a list"""
    #     self.base_test.yticklabels = 'foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_yticklabels_list_of_strings(self):
    #     """Tests checkbase errors when yticklabels is not a list of strings"""
    #     self.base_test.yticklabels = [1, 2, 3]
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_no_yticks_and_yticklabels(self):
    #     """Tests checkbase errors when yticklabels is sane but no yticks"""
    #     self.base_test.yticks = None
    #     self.base_test.yticklabels = ['foo', 'bar']
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_diff_length_yticks_and_yticklabels(self):
    #     """Tets checkbase errors when ytick and yticklabels are diff length"""
    #     self.base_test.yticks = [1, 2, 3]
    #     self.base_test.yticklabels = ['foo', 'bar']
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_xlabel_class(self):
    #     """Tests checkbase errors when xlabel is not a string"""
    #     self.base_test.xlabel = ['foo']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_ylabel_class(self):
    #     """Tests checkbase errors when ylabel is not a string"""
    #     self.base_test.ylabel = ['foo']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_show_title(self):
    #     """Tests error checking on TaxPlot is sane for the show_title class"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_title = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_title_text(self):
    #     """Tests error checking on TaxPlot is sane for the title_text class"""
    #     # Checks title handling is sane.
    #     self.base_test.title_text = ['Foo']
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_use_latex(self):
    #     """Tests error handling with use_latex class"""
    #     self.base_test.use_latex = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_latex_family_class(self):
    #     """Chesks error handling with latex_family"""
    #     self.base_test.latex_family = 'foo'
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # def test_checkbase_latex_font_class(self):
    #     """Chesks error handling with latex_font class"""
    #     self.base_test.latex_font = 3
    #     self.assertRaises(TypeError, self.base_test.checkbase)

    # def test_checkbase_priority(self):
    #     """Tests that an error occurs whe priority is not a supported value"""
    #     self.base_test.priority = 'foo'
    #     self.assertRaises(ValueError, self.base_test.checkbase)

    # # Tests check_bar
    # def test_check_barchart_bar_width_class(self):
    #     """Tests check_barchart handles bar_width class checking sanely"""
    #     self.bar_test.bar_width = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_xtick_interval_class(self):
    #     """Tests check_barchart handles xtick_interval class sanely"""
    #     self.bar_test.xtick_interval = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_width_and_interval(self):
    #     """Tests check_barchart handles a greater width sanely"""
    #     self.bar_test.bar_width = 3
    #     self.bar_test.xtick_interval = 1
    #     self.assertRaises(ValueError, self.bar_test.check_barchart)

    # def test_check_barchart_xmin_class(self):
    #     """Tests check_barchart handles xmin class sanely"""
    #     self.bar_test.xmin = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_x_font_angle(self):
    #     """Tests check_barchart handles x_font_angle sanely"""
    #     self.bar_test.x_font_angle = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)
    #     self.bar_test.x_font_angle = -25
    #     self.assertRaises(ValueError, self.bar_test.check_barchart)

    # def test_check_barchart_x_font_align(self):
    #     """Tests check_barchart handles x_font_align sanely"""
    #     self.bar_test.x_font_align = 'foo'
    #     self.assertRaises(ValueError, self.bar_test.check_barchart)

    # def test_check_barchart_show_x_labels_class(self):
    #     """Checks that check_barchart handles the show_x_labels class sanely"""
    #     self.bar_test.show_x_labels = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_show_y_labels_class(self):
    #     """Checks that check_barchart handles the show_y_labels class sanely"""
    #     self.bar_test.show_y_labels = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    #    # Tests check_piechart
    # def test_check_piechart_plot_ccw_class(self):
    #     """Tests checks_piechart handles plot_ccw class sanely"""
    #     self.pie_test.plot_ccw = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # def test_check_piechart_start_angle_class(self):
    #     """Tests checks_piechart handles start_angle class sanely"""
    #     self.pie_test.start_angle = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # def test_check_piechart_start_angle_value(self):
    #     """Tests checks_piechart handles start_angle constraints sanely"""
    #     self.pie_test.start_angle = -1
    #     self.assertRaises(ValueError, self.pie_test.check_piechart)

    # def test_check_piechart_show_labels_class(self):
    #     """Tests checks_piechart handles show_labels class sanely"""
    #     self.pie_test.show_labels = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # def test_check_piechart_numeric_labels_class(self):
    #     """Tests checks_piechart handles numeric_labels class sanely"""
    #     self.pie_test.numeric_labels = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # def test_check_piechart_labels_distance_class(self):
    #     """Tests checks_piechart errors when label_distance is not numeric"""
    #     self.pie_test.label_distance = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # # Tests check_trace
    # def test_check_trace_linestyle_class(self):
    #     """Tests check_trace errors when linestyle is not a list"""
    #     self.trace_test.linestyle = 'foo'
    #     self.assertRaises(TypeError, self.trace_test.check_trace)

    # def test_check_trace_linestyle_length(self):
    #     """Tests check_trace errors when linestyle is empty."""
    #     self.trace_test.linestyle = []
    #     self.assertRaises(ValueError, self.trace_test.check_trace)

    # def test_check_trace_linestyle_str(self):
    #     """Tests check_trace errors when the markers are not strings or None"""
    #     self.trace_test.linestyle = [1, 'foo', None]
    #     self.assertRaises(TypeError, self.trace_test.check_trace)

    # def test_check_trace_not_enough_linestyles(self):
    #     """Tests check_trace errors if there aren't lines for each dataset"""
    #     self.trace_test.linestyle = ['-', ':']
    #     self.trace_test.data = self.data
    #     self.trace_test.samples = self.samples
    #     self.trace_test.groups = self.groups
    #     self.assertRaises(ValueError, self.trace_test.check_trace)

    # def test_check_trace_linestyle_return_single(self):
    #     """Tests check_trace expands markers sanely for a single linestyle"""
    #     self.trace_test.linestyle = ['-']
    #     self.trace_test.markers = ['x']
    #     self.trace_test.check_trace()
    #     self.assertEqual(self.trace_test.linestyle, ['-', '-'])

    # def test_check_trace_markers_class(self):
    #     """Tests check_trace errors when markers is not a list"""
    #     self.trace_test.markers = 'foo'
    #     self.assertRaises(TypeError, self.trace_test.check_trace)

    # def test_check_trace_markers_length(self):
    #     """Tests check_trace errors when markers is empty."""
    #     self.trace_test.markers = []
    #     self.assertRaises(ValueError, self.trace_test.check_trace)

    # def test_check_trace_markers_str(self):
    #     """Tests check_trace errors when the markers are not strings or None"""
    #     self.trace_test.markers = [1, 'foo', None]
    #     self.assertRaises(TypeError, self.trace_test.check_trace)

    # def test_check_trace_not_enough_markers(self):
    #     """Tests check_trace errors with insuffecient maker styles"""
    #     self.trace_test.makers = ['.', 'o']
    #     self.trace_test.data = self.data
    #     self.trace_test.samples = self.samples
    #     self.trace_test.groups = self.groups
    #     self.assertRaises(ValueError, self.trace_test.check_trace)

    # def test_check_trace_markers_return(self):
    #     """Tests check_trace expands markers sanely"""
    #     self.trace_test.markers = ['x']
    #     self.trace_test.check_trace()
    #     self.assertEqual(self.trace_test.markers, ['x', 'x'])

    # # Tests check_scatter
    # def test_check_scatter_show_distribution_class(self):
    #     """Tests checks_scatter errors when show_distribution is not boolian"""
    #     self.scatter_test.show_distribution = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_show_dist_hist_class(self):
    #     """Tests checks_scatter errors when show_dist_hist is not boolian"""
    #     self.scatter_test.show_dist_hist = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_show_reg_line_class(self):
    #     """Tests checks_scatter errors when show_reg_line is not boolian"""
    #     self.scatter_test.show_reg_line = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_match_reg_line_class(self):
    #     """Tests checks_scatter errors when match_reg_line is not boolian"""
    #     self.scatter_test.match_reg_line = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_show_reg_equation_class(self):
    #     """Tests checks_scatter errors when connect_points is not boolian"""
    #     self.scatter_test.show_reg_equation = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_show_r2_class(self):
    #     """Tests checks_scatter errors when show_r2 is not boolian"""
    #     self.scatter_test.show_r2 = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_show_error_class(self):
    #     """Tests checks_scatter errors when show_error is not boolian"""
    #     self.scatter_test.show_error = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_connect_points_class(self):
    #     """Tests checks_scatter errors when connect_points is not boolian"""
    #     self.scatter_test.connect_points = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_equation_position_class(self):
    #     """Tests check_scatter error when equation_positions is not a tuple"""
    #     self.scatter_test.equation_position = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_r2_position_class(self):
    #     """Tests check_scatter error when equation_positions is not a tuple"""
    #     self.scatter_test.r2_position = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_x_axis_dims_class(self):
    #     """Tests check_scatter errors when x_axis_dims is not a tuple"""
    #     self.scatter_test.x_axis_dims = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_y_axis_dims_class(self):
    #     """Tests check_scatter errors when y_axis_dims is not a tuple"""
    #     self.scatter_test.y_axis_dims = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_marker_class(self):
    #     """Tests check_scatter errors when marker is not a reasonable class"""
    #     self.scatter_test.markers = 3
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_marker_iterable_class(self):
    #     """Tests check_scatter errors when iterable markers are not strings"""
    #     self.scatter_test.markers = [1]
    #     self.assertRaises(ValueError, self.scatter_test.check_scatter)

    # def test_check_scatter_bins_class(self):
    #     """Tests check_scatter errors when bins is not an integer"""
    #     self.scatter_test.bins = 3.1415
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_bins_value(self):
    #     """Tests check_scatter errors when bins is less than 1"""
    #     self.scatter_test.bins = -5
    #     self.assertRaises(ValueError, self.scatter_test.check_scatter)

    # def test_check_scatter_round_to_x_class(self):
    #     """Tests check_scatter errors when round_to is not an integer"""
    #     self.scatter_test.round_to_x = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_round_to_x_value(self):
    #     """Tests check_scatter errors when round_to is less than 1"""
    #     self.scatter_test.round_to_x = -5
    #     self.assertRaises(ValueError, self.scatter_test.check_scatter)

    # def test_check_scatter_round_to_y_class(self):
    #     """Tests check_scatter errors when round_to is not an integer"""
    #     self.scatter_test.round_to_y = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_round_to_y_value(self):
    #     """Tests check_scatter errors when round_to is less than 1"""
    #     self.scatter_test.round_to_y = -5
    #     self.assertRaises(ValueError, self.scatter_test.check_scatter)

    # # Tests write_object_paremeters
    # def test_write_parameters(self):
    #     """Tests an object parameters file can be generated sanely"""
    #     known_params_str = "TaxPlot:show_title\tFalse\nTaxPlot:axis_dims\t"\
    #         "(0.3, 0.3, 0.6, 0.6)\nTaxPlot:colors\tNone\nTaxPlot:fig_dims\t"\
    #         "(3, 4)\nTaxPlot:latex_family\tsans-serif\nTaxPlot:latex_font\t"\
    #         "['Helvetica', 'Arial']\nTaxPlot:legend_offset\tNone\nTaxPlot:"\
    #         "legend_properties\t{}\nTaxPlot:priority\tCURRENT\nTaxPlot:"\
    #         "save_properties\t{}\nTaxPlot:show_axes\tTrue\nTaxPlot:show_edge"\
    #         "\tFalse\nTaxPlot:show_error\tFalse\nTaxPlot:show_frame\tTrue\n"\
    #         "TaxPlot:show_legend\tFalse\nTaxPlot:title_properties\t{}\n"\
    #         "TaxPlot:title_text\tNone\nTaxPlot:use_latex\tFalse\nTaxPlot:"\
    #         "xlabel\tNone\nTaxPlot:xlim\tNone\nTaxPlot:xticklabels\tNone\n"\
    #         "TaxPlot:xticks\tNone\nTaxPlot:ylabel\tNone\nTaxPlot:ylim\tNone"\
    #         "\nTaxPlot:yticklabels\tNone\nTaxPlot:yticks\tNone"
    #     test_string = self.base_test.write_parameters()
    #     self.assertEqual(known_params_str, test_string)

    # def test_read_parameters_current_priority(self):
    #     """Tests an object parameters file can sanely keep current"""
    #     # Sets the priority on current custom values
    #     self.priority = 'CURRENT'
    #     # Loads the parameter file
    #     self.base_test.read_parameters(self.params_file)
    #     # Checks the paramaters have been set appropriate
    #     known_new_colors = 'Spectral'
    #     known_set_fig_dims = (3, 4)
    #     self.assertEqual(self.base_test.colors, known_new_colors)
    #     self.assertEqual(self.base_test.fig_dims, known_set_fig_dims)

    # def test_read_parameters_imported_priority(self):
    #     """Tests read_parameters can sanely use imported values"""
    #     # Sets the priority to imported values
    #     self.base_test.priority = 'IMPORTED'
    #     # Loads the parameter file
    #     self.base_test.read_parameters(self.params_file)
    #     # Checks the paramaters have been set appropriate
    #     known_new_colors = 'Spectral'
    #     known_set_fig_dims = (6, 4)
    #     self.assertEqual(self.base_test.colors, known_new_colors)
    #     self.assertEqual(self.base_test.fig_dims, known_set_fig_dims)

    # # Tests update_colormap
    # def test_update_colormap_class(self):
    #     """Checks update_colormap throws an error when the class is wrong"""
    #     # Colormap must be None, a string, or numpy array
    #     self.assertRaises(TypeError, self.base_test.update_colormap, ['Foo'])

    # def test_update_colormap_no_data(self):
    #     """Checks update_colormap does nothing when there is no data"""
    #     # Checks that nothing is returned when data is none.
    #     self.base_test.data = None
    #     self.base_test.colors = 'Spectral'
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      self.colormap).all())

    # def test_update_colormap_none(self):
    #     """Checks update_colormap handles sanely when colormap is None."""
    #     # Checks what happens when data is there and colormap is None.
    #     self.base_test.data = self.data
    #     self.base_test.colors = None
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      self.colormap).all())

    # def test_update_colormap_string(self):
    #     """Checks set_coloramp handles sanely when colormap is a string"""
    #     # Tests colormap assignment when colormap is a string
    #     self.base_test.colors = 'Spectral'
    #     known_colormap = translate_colorbrewer(4, 'Spectral')
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      known_colormap).all())

    # def test_update_colormap_array(self):
    #     """Checks set_coloramp handles sanely when colormap is an array"""
    #     # Tests colormap assignment when colormap is an array with fewer rows
    #     # than are supplied in data
    #     colors = array([[1, 2, 3], [1, 2, 3]])
    #     self.base_test.colors = colors
    #     self.assertRaises(ValueError, self.base_test.update_colormap)

    #     # Tests colors when a full array has been supplied
    #     colors = array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    #     self.base_test.colors = colors
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap == colors).all())

    # def test_update_colormap_show_edge(self):
    #     """Checks update_colormap can show_edge sanely"""
    #     self.base_test.show_edge = True
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__edgecolor ==
    #                      zeros(self.colormap.shape)).all())

    # def test_update_colormap_hide_edge(self):
    #     self.base_test.show_edge = False
    #     self.base_test.update_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__edgecolor ==
    #                      self.colormap).all())

    # # Tests update_dimensions
    # def test_update_dimensions(self):
    #     """Checks update_dimensions works correctly with sane inputs"""
    #      # Sets up fig_dims and axis_dis
    #     self.base_test.fig_dims = (4, 6)
    #     self.base_test.axis_dims = (0.2, 0.2, 0.7, 0.7)
    #     self.base_test.update_dimensions()
    #     axis_bounds = self.base_test._TaxPlot__axes.get_position().bounds
    #     # Checks the figures dimensions are sane
    #     assert_array_equal(array(list(self.base_test.fig_dims)),
    #                        self.base_test._TaxPlot__fig.get_size_inches())
    #     assert_almost_equal(self.base_test.axis_dims, axis_bounds, decimal=4)

    # # Tests update_scatter_dimensions
    # def test_update_scatter_dimensions_no_show(self):
    #     """Checks update_scatter_dimensions handles a single axis sanely"""
    #     # Turns off show_distribution
    #     self.scatter_test.show_distribution = False
    #     # Updates the axes
    #     self.scatter_test.update_scatter_dimensions()
    #     # Checks the figure is sane
    #     assert_array_equal(array(list(self.scatter_test.fig_dims)),
    #                        self.scatter_test._TaxPlot__fig.get_size_inches())
    #     # Checks the figure and main axes are sane
    #     self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
    #     m_ax_bounds = self.scatter_test._TaxPlot__axes.get_position().bounds
    #     assert_almost_equal(self.scatter_test.axis_dims, m_ax_bounds,
    #                         decimal=4)
    #     self.assertTrue(self.scatter_test._TaxPlot__axes.get_visible())
    #     # Checks the secondary axes
    #     self.assertEqual(self.scatter_test._ScatterPlot__x_axes, None)
    #     self.assertEqual(self.scatter_test._ScatterPlot__y_axes, None)

    # def test_update_scatter_dimensions_show_dist(self):
    #     """Checks update_scatter_dimensions handles showing axes sanely"""
    #     # Turns on the distribution
    #     self.scatter_test.show_distribution = True
    #     # Updates the dimensions
    #     self.scatter_test.update_scatter_dimensions()
    #     # Checks the figure is sane
    #     assert_array_equal(array(list(self.scatter_test.fig_dims)),
    #                        self.scatter_test._TaxPlot__fig.get_size_inches())
    #     # Checks the figure and main axes are sane
    #     self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
    #     m_ax_bounds = self.scatter_test._TaxPlot__axes.get_position().bounds
    #     assert_almost_equal(self.scatter_test.axis_dims, m_ax_bounds,
    #                         decimal=4)
    #     self.assertTrue(self.scatter_test._TaxPlot__axes.get_visible())
    #     # Checks the x-axis is sane
    #     self.assertTrue(isinstance(self.scatter_test._ScatterPlot__x_axes,
    #                                plt.Axes))
    #     x_ax_bounds = \
    #         self.scatter_test._ScatterPlot__x_axes.get_position().bounds
    #     assert_almost_equal(self.scatter_test.x_axis_dims, x_ax_bounds,
    #                         decimal=4)
    #     self.assertTrue(self.scatter_test._ScatterPlot__x_axes.get_visible())
    #      # Checks the y-axis is sane
    #     self.assertTrue(isinstance(self.scatter_test._ScatterPlot__y_axes,
    #                                plt.Axes))
    #     y_ax_bounds = \
    #         self.scatter_test._ScatterPlot__y_axes.get_position().bounds
    #     assert_almost_equal(self.scatter_test.y_axis_dims, y_ax_bounds,
    #                         decimal=4)
    #     self.assertTrue(self.scatter_test._ScatterPlot__y_axes.get_visible())

    # # Tests set_filepath
    # def test_set_filepath_no_filepath(self):
    #     """Checks TaxPlot can sanely update the filepath when None is given"""
    #     # Tries for an instance where filepath is none
    #     filename = None
    #     known_filename = None
    #     known_filepath = None
    #     known_filetype = None
    #     self.base_test.set_filepath(filename)
    #     self.assertEqual(self.base_test._TaxPlot__filepath, known_filepath)
    #     self.assertEqual(self.base_test._TaxPlot__filename, known_filename)
    #     self.assertEqual(self.base_test._TaxPlot__filetype, known_filetype)

    # def test_set_filepath_string(self):
    #     """Checks TaxPlot can sanely update a string filepath"""
    #     # Tries for an instance where filepath is a string
    #     filename = '$HOME/test/test.txt'
    #     known_filepath = '$HOME/test'
    #     known_filename = 'test.txt'
    #     known_filetype = 'TXT'
    #     self.base_test.set_filepath(filename)
    #     self.assertEqual(self.base_test._TaxPlot__filepath, known_filepath)
    #     self.assertEqual(self.base_test._TaxPlot__filename, known_filename)
    #     self.assertEqual(self.base_test._TaxPlot__filetype, known_filetype)
    #     # Tries an instance where the filepath is not a string
    #     self.assertRaises(TypeError, self.base_test.set_filepath, 3)

    # # Tests set_font
    # def test_set_font_unsupported_type(self):
    #     """Checks an error is thrown when the font type is not sane"""
    #     font_type = 'Foo'
    #     font_object = FontProperties()
    #     self.assertRaises(ValueError, self.base_test.set_font, font_type,
    #                       font_object)

    # def test_set_font_unsupported_object(self):
    #     """Checks an error is thrown when the font object is not sane"""
    #     font_type = 'title'
    #     font_object = 'Helvetica'
    #     self.assertRaises(TypeError, self.base_test.set_font, font_type,
    #                       font_object)

    # def test_set_font(self):
    #     """Checks that a font can be set sanely using set_font"""
    #     # Sets up the test values
    #     font_type = 'title'
    #     font_object = FontProperties(family='cursive', size=25)
    #     self.base_test.set_font(font_type, font_object)
    #     # Checks the known font and the set font are the same
    #     test = self.base_test._TaxPlot__font_set[font_type]
    #     self.assertTrue(isinstance(test, FontProperties))
    #     self.assertEqual(test.get_family(), font_object.get_family())
    #     self.assertEqual(test.get_name(), font_object.get_name())
    #     self.assertEqual(test.get_size(), font_object.get_size())
    #     self.assertEqual(test.get_slant(), font_object.get_slant())
    #     self.assertEqual(test.get_stretch(), font_object.get_stretch())
    #     self.assertEqual(test.get_style(), font_object.get_style())
    #     self.assertEqual(test.get_variant(), font_object.get_variant())
    #     self.assertEqual(test.get_weight(), font_object.get_weight())

    # # Tests the set of "get" functions
    # def test_get_colormap(self):
    #     """Checks the colormap can be sanely returned"""
    #     known = self.base_test._TaxPlot__colormap
    #     test = self.base_test.get_colormap()
    #     self.assertTrue((known == test).all())

    # def test_get_filepath_no_filetype(self):
    #     """Checks the filepath can be returned sanely when none was supplied"""
    #     known = None
    #     self.base_test._TaxPlot__filetype
    #     test = self.base_test.get_filepath()
    #     self.assertEqual(known, test)

    # def test_get_filepath_string(self):
    #     """Checks the filepath can be returned sanely when its a string"""
    #     known = '/Users/jwdebelius/Desktop/test.pdf'
    #     self.base_test.set_filepath(known)
    #     test = self.base_test.get_filepath()
    #     self.assertEqual(test, known)

    # def test_get_font_unsupported(self):
    #     """Checks get_font raises an error when the fonttype is unsupported"""
    #     self.assertRaises(ValueError, self.base_test.get_font, 'foo')

    # def test_get_font_supported(self):
    #     """Checks get_font returns the correct font when type is specified."""
    #     font = 'title'
    #     known = self.base_test._TaxPlot__font_set[font]
    #     test = self.base_test.get_font(font)
    #     self.assertTrue(isinstance(test, FontProperties))
    #     self.assertEqual(test.get_family(), known.get_family())
    #     self.assertEqual(test.get_name(), known.get_name())
    #     self.assertEqual(test.get_size(), known.get_size())
    #     self.assertEqual(test.get_slant(), known.get_slant())
    #     self.assertEqual(test.get_stretch(), known.get_stretch())
    #     self.assertEqual(test.get_style(), known.get_style())
    #     self.assertEqual(test.get_variant(), known.get_variant())
    #     self.assertEqual(test.get_weight(), known.get_weight())

    # # Calculates the smoothed range
    # def calculate_smoothed_range_custom_x(self):
    #     """Tests that calculate_smoothed_range ignores custom xlims"""
    #     # Sets a custom range
    #     known_range = [0, 300]
    #     self.scatter_test.xlim = known_range
    #     # Calculates the best range for the data
    #     self.scatter_test.calculate_smoothed_range()
    #     # Checks the known matches the test value
    #     self.assertEqual(known_range, self.scatter_test.xlim)

    # def calculate_smoothed_range_custom_y(self):
    #     """Tests calculate_smoothed_range ignores custom ylim"""
    #     # Sets a custom range
    #     known_range = [0, 300]
    #     self.scatter_test.ylim = known_range
    #     # Calculates the best range for the data
    #     self.scatter_test.calculate_smoothed_range()
    #     # Checks the known matches the test value
    #     self.assertEqual(known_range, self.scatter_test.ylim)

    # def calculate_smoothed_range_default_x(self):
    #     """Tests calculate_smoothed_range can calculate a sane xlim"""
    #     # Sets up the known range for the default x parameters
    #     known_range = [0, 25]
    #     self.scatter_test.xlim = None
    #     # Calculates the best range for hte data
    #     self.scatter_test.calculate_smoothed_range()
    #     # Checks the known matches the test value
    #     self.assertEqual(known_range, self.scatter_test.xlim)

    # def calculate_smoothed_range_default_y(self):
    #     """Tests calculate_smoothed_range can calculate a sane y range"""
    #     # Sets up the known range for the default x parameters
    #     known_range = [4, 18]
    #     self.scatter_test.xlim = None
    #     self.scatter_test.round_to_y = 2
    #     # Calculates the best range for hte data
    #     self.scatter_test.calculate_smoothed_range()
    #     # Checks the known matches the test value
    #     self.assertEqual(known_range, self.scatter_test.ylim)

    # # Tests calculate_distribution
    # def test_calculate_distribution_trace(self):
    #     """Tests calculate_distribution bins data correctly for trace plots"""
    #     # Sets up the known (default) values for the distribution
    #     x_dist = array([0.16, 0.20, 0.20, 0.20, 0.24])
    #     x_bins = array([2.5,  7.5, 12.5, 17.5, 22.5])
    #     y_dist = [array([0.20, 0.28, 0.24, 0.24, 0.04])]
    #     y_bins = [array([6.5,  9.5, 12.5, 15.5, 18.5])]
    #     # Sets a custom number of bins
    #     self.scatter_test.data = self.dep[:, 0]
    #     self.scatter_test.error = None
    #     self.scatter_test.samples = ['Venom Activity']
    #     self.scatter_test.ylim = [5, 20]
    #     self.scatter_test.bins = 5
    #     self.scatter_test.show_dist_hist = False
    #     # Calculates the distribution
    #     self.scatter_test.calculate_distribution()
    #     # Checks the x_axis related data
    #     assert_almost_equal(self.scatter_test._ScatterPlot__x_dist, x_dist,
    #                         decimal=5)
    #     assert_array_equal(self.scatter_test._ScatterPlot__x_bins, x_bins)
    #     # Checks the y_axis related data
    #     for idx, a in enumerate(self.scatter_test._ScatterPlot__y_dist):
    #         b = self.scatter_test._ScatterPlot__y_bins[idx]
    #         assert_almost_equal(y_dist[idx], a, decimal=5)
    #         assert_array_equal(y_bins[idx], b)

    # def test_calculate_distribution_histogram(self):
    #     """Tests calculate_distribution bins data correctly for histograms"""
    #     # Sets up the known (default) values for the distribution
    #     x_dist = array([0.16, 0.20, 0.20, 0.20, 0.24])
    #     x_bins = array([0, 5, 10, 15, 20])
    #     y_dist = [array([0.00,  0.00,  0.40,  0.40,  0.20]),
    #               array([0.08,  0.36,  0.40,  0.16,  0.00])]
    #     y_bins = [array([-5., 0., 5., 10., 15.]),
    #               array([-5., 0., 5., 10., 15.])]
    #     # Sets up a custom scatter object
    #     self.scatter_test.ylim = [-5, 20]
    #     self.scatter_test.bins = 5
    #     self.scatter_test.show_dist_hist = True
    #     # Sets a custom number of bins and turns off normalization
    #     # Calculates the distribution
    #     self.scatter_test.calculate_distribution()
    #     # Checks the x_axis related data
    #     assert_almost_equal(self.scatter_test._ScatterPlot__x_dist, x_dist,
    #                         decimal=5)
    #     assert_array_equal(self.scatter_test._ScatterPlot__x_bins, x_bins)
    #     # Checks the y_axis related data
    #     for idx, a in enumerate(self.scatter_test._ScatterPlot__y_dist):
    #         b = self.scatter_test._ScatterPlot__y_bins[idx]
    #         assert_almost_equal(y_dist[idx], a, decimal=5)
    #         assert_array_equal(y_bins[idx], b)

    # # Test calculate_regression
    # def test_calculate_regression_vector_data(self):
    #     """Tests a regression can be calculated sanely for vector data"""
    #     # Sets up the known values
    #     x_reg = arange(0, 25+0.5, 0.5)
    #     y_reg = [array([05.079,  5.326,  5.573,  5.821,  6.068,
    #                     06.315,  6.562,  6.810,  7.057,  7.304,
    #                     07.552,  7.799,  8.046,  8.294,  8.541,
    #                     08.788,  9.035,  9.283,  9.530,  9.777,
    #                     10.025, 10.272, 10.519, 10.767, 11.014,
    #                     11.261, 11.508, 11.756, 12.003, 12.250,
    #                     12.498, 12.745, 12.992, 13.240, 13.487,
    #                     13.734, 13.981, 14.229, 14.476, 14.723,
    #                     14.971, 15.218, 15.465, 15.713, 15.960,
    #                     16.207, 16.454, 16.702, 16.949, 17.196,
    #                     17.444])]
    #     stats = (0.494599,  5.07865, 0.999454, 1.44e-35, 0.00341)
    #     # Sets up the test values
    #     self.scatter_test.data = self.dep[:, 0]
    #     self.scatter_test.error = None
    #     self.scatter_test.samples = ['Venom Activity']
    #     # Calculates the regression
    #     self.scatter_test.calculate_regression()
    #     # Tests the results of the regression
    #     assert_almost_equal(self.scatter_test._ScatterPlot__x_reg, x_reg,
    #                         decimal=3)
    #     assert_almost_equal(self.scatter_test._ScatterPlot__y_reg[0], y_reg[0],
    #                         decimal=3)
    #     (m, b, r, p, s) = self.scatter_test._ScatterPlot__reg_stats[0]
    #     assert_almost_equal(stats[0], m, decimal=5)
    #     assert_almost_equal(stats[1], b, decimal=5)
    #     assert_almost_equal(stats[2], r, decimal=5)
    #     assert_almost_equal(log10(stats[3]), log10(p), decimal=2)
    #     assert_almost_equal(stats[4], s, decimal=5)

    # def test_calculate_regression_array_data(self):
    #     """Tests a regression can be calculated sanely for array data"""
    #     x_reg = arange(0, 25+0.5, 0.5)
    #     y_reg = [array([05.079,  5.326,  5.573,  5.821,  6.068,
    #                     06.315,  6.562,  6.810,  7.057,  7.304,
    #                     07.552,  7.799,  8.046,  8.294,  8.541,
    #                     08.788,  9.035,  9.283,  9.530,  9.777,
    #                     10.025, 10.272, 10.519, 10.767, 11.014,
    #                     11.261, 11.508, 11.756, 12.003, 12.250,
    #                     12.498, 12.745, 12.992, 13.240, 13.487,
    #                     13.734, 13.981, 14.229, 14.476, 14.723,
    #                     14.971, 15.218, 15.465, 15.713, 15.960,
    #                     16.207, 16.454, 16.702, 16.949, 17.196,
    #                     17.444]),
    #              array([11.573, 11.327, 11.081, 10.835, 10.589,
    #                     10.343, 10.097,  9.851,  9.605,  9.359,
    #                     09.113,  8.867,  8.621,  8.375,  8.129,
    #                     07.883,  7.637,  7.392,  7.146,  6.900,
    #                     06.654,  6.408,  6.162,  5.916,  5.670,
    #                     05.424,  5.178,  4.932,  4.686,  4.440,
    #                     04.194,  3.948,  3.702,  3.456,  3.210,
    #                     02.964,  2.718,  2.472,  2.226,  1.980,
    #                     01.735,  1.489,  1.243,  0.997,  0.751,
    #                     00.505,  0.259,  0.013, -0.233, -0.479,
    #                     -0.725])]
    #     stats = [(00.494599,  5.07865,  0.999454, 1.44e-35, 0.00341),
    #              (-0.491915, 11.57282, -0.967132, 3.56e-15, 0.026968)]
    #     # Calculates the regression
    #     self.scatter_test.calculate_regression()
    #     # Tests the results of the regression
    #     assert_almost_equal(self.scatter_test._ScatterPlot__x_reg, x_reg,
    #                         decimal=3)
    #     # Test the y_regressions and stats
    #     for idx, stat in enumerate(self.scatter_test._ScatterPlot__reg_stats):
    #         k_stats = stats[idx]
    #         (m, b, r, p, s) = stat
    #         assert_almost_equal(k_stats[0], m, decimal=5)
    #         assert_almost_equal(k_stats[1], b, decimal=5)
    #         assert_almost_equal(k_stats[2], r, decimal=5)
    #         assert_almost_equal(log10(k_stats[3]), log10(p), decimal=2)
    #         assert_almost_equal(k_stats[4], s, decimal=5)
    #         reg = self.scatter_test._ScatterPlot__y_reg[idx]
    #         assert_almost_equal(reg, y_reg[idx], decimal=3)

    # # Tests save_figure.
    # # Other tests are included in the render_X tests (the figures get saved)
    # def test_save_fig_no_filepath(self):
    #     """Checks that a file is not generated when the filetype is None."""
    #     test_filename = pjoin(TEST_DIR, 'files/file_does_not_exits.file')
    #     self.base_test.set_filepath(test_filename)
    #     self.base_test._TaxPlot__filetype = None
    #     self.base_test.save_figure()
    #     self.assertFalse(exists(test_filename))

    def test_save_figure_filepath(self):
        """Checks that a file is generated adn the file contains axes"""
        test_filename = pjoin(TEST_DIR, 'files/save_test.pdf')
        self.base_test.set_filepath(test_filename)
        self.base_test.save_figure()
    # # Tests rendering which does not result in figures
    # def test_render_legend_show_false(self):
    #     """Tests render_legend handles sanely when show_legend is false"""
    #     # Renders the bar chart so patches are there
    #     self.bar_test.show_legend = False
    #     self.bar_test.render_barchart()
    #     # Sets up the known value of legend
    #     known_legend = None
    #     # Tests that no legend is created
    #     self.bar_test.render_legend()
    #     self.assertEqual(known_legend, self.bar_test._TaxPlot__legend)

    # def test_render_legend_no_patches(self):
    #     """Tests render_legend handles sanely when no data has been plotted"""
    #     # Renders the bar chart so patches are there
    #     self.bar_test.show_legend = True
    #     self.bar_test._TaxPlot__patches = []
    #     # Sets up the known value of legend
    #     known_legend = None
    #     # Tests that no legend is created
    #     self.bar_test.render_legend()
    #     self.assertEqual(known_legend, self.bar_test._TaxPlot__legend)

    # def test_render_title_no_axes(self):
    #     """Checks render_title handles sanely with no axes"""
    #     self.base_test.show_title = True
    #     self.base_test._TaxPlot__axes = None
    #     known_title = None
    #     self.base_test.render_title()
    #     self.assertEqual(known_title, self.base_test._TaxPlot__title)

    # def test_render_title_show_false(self):
    #     """Checks render_title handles sanely when show_title is false"""
    #     self.base_test.show_title = False
    #     known_title = None
    #     self.base_test.render_title()
    #     self.assertEqual(known_title, self.base_test._TaxPlot__title)

    # def test_render_piechart_warning(self):
    #     """Checks a warning is triggered when data is a matrix"""
    #     with catch_warnings(record=True) as w:
    #         # Triggers the warnings
    #         test = PieChart(fig_dims=self.fig_dims,
    #                         axis_dims=self.axis_dims,
    #                         data=self.data,
    #                         groups=self.groups,
    #                         samples=self.samples)
    #         test.render_piechart()
    #         # Verifies somethings about the warning
    #         self.assertTrue(len(w) == 1)
    #         self.assertTrue(issubclass(w[-1].category, UserWarning))

    # # Tests rendering which results in figures
    # def test_render_legend_and_barchart_error(self):
    #     """Checks the barchart can be rendered sanely with errorbars"""
    #     # Sets up the figure to render
    #     test = BarChart(fig_dims=(6, 3), axis_dims=(0.2, 0.35, 0.4, 0.5),
    #                     data=self.data, groups=self.groups,
    #                     samples=self.samples, error=self.error,
    #                     colors='Spectral', show_legend=True,
    #                     legend_offset=(1.9, 0.9), show_error=True,
    #                     ylim=[0, 1])
    #     test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     test.set_filepath(test_filename)
    #     test.render_barchart()
    #     self.assertTrue(isinstance(test._TaxPlot__legend, Legend))
    #     self.assertTrue(isinstance(self.bar_test._TaxPlot__patches[0],
    #                                Rectangle))
    #     test.save_figure()
    #     # Sets up the known filestring
    #     known_file = open(pjoin(TEST_DIR, 'files/known_render_leg_err.svg'),
    #                       'U')
    #     known_fig = known_file.read()
    #     known_file.close()
    #     # Reads in the test figure
    #     test_file = open(test_filename, 'U')
    #     test_fig = test_file.read()
    #     test_file.close()
    #     self.assertEqual(known_fig, test_fig)
    #     # Removes the test figure
    #     # remove(test_filename)

    # def test_render_title_and_bar_no_error(self):
    #     """Test render_title respond sanely and plot_barchart_no_error"""
    #     # Generates a bar chart using the test data
    #     self.bar_test.title_text = 'HP Villians'
    #     self.bar_test.show_title = True
    #     self.bar_test.axis_dims = (0.25, 0.275, 0.625, 0.6)
    #     self.bar_test.render_barchart()
    #     # Checks that there is a title object added
    #     self.assertTrue(isinstance(self.bar_test._TaxPlot__title, Text))
    #     self.assertTrue(isinstance(self.bar_test._TaxPlot__patches[0],
    #                                Rectangle))
    #     # Saves the figure
    #     test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     self.bar_test.set_filepath(test_filename)
    #     self.bar_test.save_figure()
    #     # Sets up the known filestring
    #     known_file = open(pjoin(TEST_DIR, 'files/known_render_title.svg'), 'U')
    #     known_fig = known_file.read()
    #     known_file.close()
    #     # Reads in the test figure
    #     test_file = open(test_filename, 'U')
    #     test_fig = test_file.read()
    #     test_file.close()
    #     self.assertEqual(known_fig, test_fig)
    #     # Removes the test figure
    #     remove(test_filename)

    # def test_render_piechart(self):
    #     """Tests that render_piechart returns sanely"""
    #     # Sets up the plotting properties
    #     self.pie_test.colors = 'RdPu'
    #     self.pie_test.plot_ccw = True
    #     self.pie_test.use_xkcd = True
    #     # Renders the piechart
    #     self.pie_test.render_piechart()
    #     # Checks a piechart object was rendered
    #     self.assertTrue(isinstance(self.pie_test._TaxPlot__patches[0], Wedge))
    #     self.assertTrue(isinstance(self.pie_test._PieChart__labels[0], Text))
    #     self.assertTrue(self.pie_test._TaxPlot__axes.get_visible())
        # Saves the figure
        # test_filename = pjoin(TEST_DIR, 'files/test.pdf')
        # self.pie_test.set_filepath(test_filename)
        # self.pie_test.save_figure()
    # #     # Sets up the known filestring
    # #     known_file = open(pjoin(TEST_DIR, 'files/known_piechart.svg'), 'U')
    # #     known_fig = known_file.read()
    # #     known_file.close()
    # #     # Reads in the test figure
    # #     test_file = open(test_filename, 'U')
    # #     test_fig = test_file.read()
    # #     test_file.close()
    # #     self.assertEqual(known_fig, test_fig)
    # #     # Removes the test figure
    # #     remove(test_filename)

    # def test_render_scatter_vector_with_errorbars(self):
    #     """Tests render_scatter can sanely handle vectors and error bars"""
    #     # Sets up plotting properties
    #     self.scatter_test.data = self.dep[:, 0]
    #     self.scatter_test.erorr = self.err[:, 0]
    #     self.samples = ['Gryffendors']
    #     self.show_dist = False
    #     self.axis_dims = (0.25, 0.25, 0.5, 0.5)
    #     # Creates the figure
    #     self.scatter_test.render_scatterplot()
    #     self.scatter_test._TaxPlot__axes.get_visible()
        # self.assertEqual(self.scatter_test.)
        # # Saves the figure
        # test_filename = pjoin(TEST_DIR, 'files/test.pdf')
        # self.scatter_test.set_filepath(test_filename)
        # self.scatter_test.save_figure()

    # def test_render_trace(self):
    #     """Tests that render_trace runs sanely"""
    #     self.trace_test.data = self.dep
    #     self.trace_test.groups = self.ind
    #     self.trace_test.samples = ['Gryffendors', 'Slythrins']
    #     self.trace_test.colors = array([[1, 0.0, 0],
    #                                     [0, 0.5, 0]])
    #     self.trace_test.render_trace()
    #     test_filename = pjoin(TEST_DIR, 'files/trace.pdf')
    #     self.trace_test.set_filepath(test_filename)
    #     self.trace_test.save_figure()

if __name__ == '__main__':
    main()
