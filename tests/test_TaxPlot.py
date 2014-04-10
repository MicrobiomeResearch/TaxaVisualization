# #!/usr/bin/env python
# test_build_cat_table.py

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

from unittest import TestCase, main
from os import remove
from os.path import realpath, dirname, join as pjoin, exists
from warnings import (filterwarnings, catch_warnings)
from numpy import array, zeros, arange
from numpy.testing import assert_array_equal, assert_almost_equal
from matplotlib import use
use('agg', warn=False)
from matplotlib.legend import Legend
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from TaxPlot import TaxPlot, BarChart, PieChart, ScatterPlot
from americangut.make_phyla_plots import translate_colorbrewer

# Determines the location fo the reference files
TEST_DIR = dirname(realpath(__file__))

# Stops warnings from being triggered
filterwarnings("always")


class TestTaxPlot(TestCase):

    def setUp(self):
        """Sets up variables for testing"""
        # Sets up distributions
        self.cont_ind = array([01.000,  2.000,  3.000,  4.000,  5.000,
                               06.000,  7.000,  8.000,  9.000, 10.000,
                               11.000, 12.000, 13.000, 14.000, 15.000,
                               16.000, 17.000, 18.000, 19.000, 20.000,
                               21.000, 22.000, 23.000, 24.000, 25.000])
        self.cont_dep = array([5.667,  5.960,  6.501,  7.273,   7.346,
                               08.075,  8.773,  8.799,  9.543, 09.904,
                               10.634, 10.946, 11.545, 12.114, 12.542,
                               12.927, 13.579, 13.935, 14.486,  14.945,
                               15.300, 15.895, 16.623, 16.940,  17.459])
        self.cont_err = array([00.320,  0.465,  0.674,  0.445,  0.433,
                               00.498,  0.552,  0.344,  0.758,  0.542,
                               00.594,  0.378,  0.390,  0.441,  0.378,
                               00.503,  0.446,  0.445,  0.545,  0.537,
                               00.456,  0.361,  0.562,  0.486,  0.506])
        self.data = array([[0.1, 0.2, 0.3],
                           [0.2, 0.3, 0.4],
                           [0.3, 0.4, 0.1],
                           [0.4, 0.1, 0.2]])
        self.samples = ['Harry', 'Ron', 'Hermione']
        self.groups = ['Snape', 'D_Malfoy', 'Umbridge', 'Voldemort']
        self.meta = {}
        self.error = array([[0.100, 0.100, 0.100],
                            [0.050, 0.050, 0.050],
                            [0.010, 0.010, 0.010],
                            [0.005, 0.005, 0.005]])
        self.fig_dims = (3, 4)
        self.axis_dims = (0.3, 0.3, 0.6, 0.6)
        self.colormap = array([[0.00, 0.00, 0.00],
                               [0.25, 0.25, 0.25],
                               [0.50, 0.50, 0.50],
                               [0.75, 0.75, 0.75]])
        self.base_test = TaxPlot(fig_dims=self.fig_dims,
                                 axis_dims=self.axis_dims,
                                 data=self.data,
                                 groups=self.groups,
                                 samples=self.samples,
                                 error=self.error)
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
                                    'legend_properties', 'match_legend',
                                    '_TaxPlot__patches',
                                    '_TaxPlot__legend', 'show_axes',
                                    'show_frame', 'axis_properties',
                                    'show_title', 'title_text',
                                    'title_properties', '_TaxPlot__title',
                                    'use_latex', 'latex_family', 'latex_font',
                                    '_TaxPlot__font_set',
                                    '_TaxPlot__properties'])
        self.bar_test = BarChart(fig_dims=self.fig_dims,
                                 axis_dims=self.axis_dims,
                                 data=self.data,
                                 groups=self.groups,
                                 samples=self.samples,
                                 error=self.error)
        self.bar_properties = set(['bar_width', 'x_min',
                                   'x_tick_interval', 'x_font_angle',
                                   'x_font_align', 'show_x_labels',
                                   'show_y_labels', '_BarChart__bar_left',
                                   '_BarChart__all_faces'
                                   ]).union(self.base_properties)
        self.pie_test = PieChart(fig_dims=(3, 3),
                                 axis_dims=self.axis_dims,
                                 data=self.data[:, 1],
                                 groups=self.groups,
                                 samples=self.samples)
        self.pie_properties = set(['plot_ccw', 'start_angle', 'axis_lims',
                                   'show_labels', 'numeric_labels',
                                   'label_distance', '_PieChart__labels',
                                   ]).union(self.base_properties)
        self.scatter_test = ScatterPlot(data=self.cont_ind,
                                        groups=self.cont_dep,
                                        samples=['Basilesk Venom'])
        self.scatter_properties = set(['show_distribution',
                                       'show_dist_hist',
                                       'show_reg_line', 'match_reg_line',
                                       'show_reg_equation', 'equation_postion',
                                       'show_r2', 'r2_position',
                                       'connect_points', 'x_axis_dimensions',
                                       'y_axis_dimensions', 'markers',
                                       '_ScatterPlot__x_axes',
                                       '_ScatterPlot__y_axes',
                                       '_ScatterPlot__x_dist',
                                       '_ScatterPlot__y_dist',
                                       ]).union(self.base_properties)
        # self.box_test = None
        # self.box_properties = None

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
    #     match_legend = True
    #     show_axes = True
    #     show_frame = True
    #     axis_properties = {}
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
    #     self.assertEqual(self.base_test.match_legend, match_legend)
    #     self.assertEqual(self.base_test.show_axes, show_axes)
    #     self.assertEqual(self.base_test.show_frame, show_frame)
    #     self.assertEqual(self.base_test.axis_properties, axis_properties)
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
    #     x_min = -0.5
    #     x_tick_interval = 1.0
    #     x_font_angle = 45
    #     x_font_align = 'right'
    #     # Compares the properties to the known
    #     self.assertEqual(self.bar_test.bar_width, bar_width)
    #     self.assertEqual(self.bar_test.x_min, x_min)
    #     self.assertEqual(self.bar_test.x_tick_interval, x_tick_interval)
    #     self.assertEqual(self.bar_test.x_font_angle, x_font_angle)
    #     self.assertEqual(self.bar_test.x_font_align, x_font_align)
    #     self.assertEqual(self.bar_properties,
    #                      self.bar_test._TaxPlot__properties)

    # def test_pie_init(self):
    #     """Checks PieChart Object initialize correctly"""
    #     # Sets up known default values
    #     plot_ccw = False
    #     start_angle = 90
    #     axis_lims = [-1.1, 1.1]
    #     show_labels = False
    #     numeric_labels = False
    #     label_distance = 1.1
    #     # Compares the properties to the known
    #     self.assertEqual(self.pie_test.plot_ccw, plot_ccw)
    #     self.assertEqual(self.pie_test.start_angle, start_angle)
    #     self.assertEqual(self.pie_test.axis_lims, axis_lims)
    #     self.assertEqual(self.pie_test.show_labels, show_labels)
    #     self.assertEqual(self.pie_test.numeric_labels, numeric_labels)
    #     self.assertEqual(self.pie_test.label_distance, label_distance)
    #     self.assertEqual(self.pie_properties,
    #                      self.pie_test._TaxPlot__properties)

    # def test_scatter_init(self):
    #     """Checks a ScatterChart object initializes sanely"""
    #     # Sets up the known default values
    #     show_distribution = True
    #     show_dist_hist = False
    #     show_reg_line = False
    #     match_reg_line = False
    #     show_reg_equation = False
    #     show_r2 = False
    #     connect_points = False
    #     equation_postion = ()
    #     r2_position = ()
    #     x_axis_dimensions = (0.09375, 0.66667, 0.62500, 0.25000)
    #     y_axis_dimensions = (0.75000, 0.12500, 0.18750, 0.50000)
    #     markers = ['x', 'o', '.', '^', 's', '*']
    #     # Compares the known default properties to the instance properties
    #     self.assertEqual(self.scatter_test.show_distribution,
    #                      show_distribution)
    #     self.assertEqual(self.scatter_test.show_dist_hist,
    #                      show_dist_hist)
    #     self.assertEqual(self.scatter_test.show_reg_line, show_reg_line)
    #     self.assertEqual(self.scatter_test.match_reg_line, match_reg_line)
    #     self.assertEqual(self.scatter_test.show_reg_equation,
    #                      show_reg_equation)
    #     self.assertEqual(self.scatter_test.equation_postion, equation_postion)
    #     self.assertEqual(self.scatter_test.show_r2, show_r2)
    #     self.assertEqual(self.scatter_test.r2_position, r2_position)
    #     self.assertEqual(self.scatter_test.connect_points, connect_points)
    #     self.assertEqual(self.scatter_test.x_axis_dimensions,
    #                      x_axis_dimensions)
    #     self.assertEqual(self.scatter_test.y_axis_dimensions,
    #                      y_axis_dimensions)
    #     self.assertEqual(self.scatter_test.markers, markers)
    #     self.assertEqual(self.scatter_properties,
    #                      self.scatter_test._TaxPlot__properties)

    # def test_boxplot_init(self):
    #     """Checks a BoxPlot instance initializes sanely"""
    #     pass

    # def test_add_attributes(self):
    #     """Checks that keyword argument attributes can be added correctly"""
    #     show_legend = True
    #     self.base_test.add_attributes(show_legend=show_legend)
    #     self.assertEqual(self.base_test.show_legend, show_legend)

    # # Tests check_base
    # def test_check_base_show_error_class(self):
    #     """Tests check_base throws an error when show_error is not a bool"""
    #     self.base_test.show_error = 'foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_save_properties_class(self):
    #     """"Checks the save_properties class handling is sane"""
    #     self.base_test.save_properties = 'foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_fig_classs(self):
    #     """Checks that an error is called when fig_dims class is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = 'foo'
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_heck_base_fig_dim_classs(self):
    #     """Checks that an error is called when fig_dims class is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = ('foo')
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_set_dimensions_num_fig_dims(self):
    #     """Checks that an error is called when fig_dims length is wrong"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.fig_dims = (1, 2, 3)
    #     # Checks an error is thrown
    #     self.assertRaises(ValueError, self.base_test.check_base)

    # def test_check_base_axis_class(self):
    #     """Checks an error is called when axis_dims is of the wrong class"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.axis_dims = 'foo'
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_axis_dims_error(self):
    #     """Checks an error is called when axis_dims is of the wrong class"""
    #     # Sets up fig_dims and axis_dims
    #     self.base_test.axis_dims = (1, 2, 3)
    #     # Checks an error is thrown
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_show_edge_class(self):
    #     """Checks the show_edge class checking is sane"""
    #     self.base_test.show_edge = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_show_legend_class(self):
    #     """Checks the show_edge class checking is sane"""
    #     self.base_test.show_legend = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_legend_properties(self):
    #     """Checks sanity of legend_properties class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.legend_properties = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_legend_offset(self):
    #     """Checks sanity of legend_offset checking"""
    #     # Checks the show_title handling is sane
    #     self.base_test.legend_offset = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)
    #     self.base_test.legend_offset = ('Foo', 'Bar', 'Cat')
    #     self.assertRaises(ValueError, self.base_test.check_base)

    # def test_check_base_match_legend_class(self):
    #     """Checks that check_barchart handles the match_legend class sanely"""
    #     self.base_test.match_legend = 'foo'
    #     self.assertRaises(TypeError, self.base_test.check_barchart)

    # def test_check_base_show_axes(self):
    #     """Checks sanity of show_axes class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_axes = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_show_frame(self):
    #     """Checks sanity of show_frame class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_frame = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_axis_properties(self):
    #     """Checks sanity of x_axis_properties class check"""
    #     # Checks the show_title handling is sane
    #     self.base_test.axis_properties = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_show_title(self):
    #     """Tests error checking on TaxPlot is sane for the show_title class"""
    #     # Checks the show_title handling is sane
    #     self.base_test.show_title = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_title_text(self):
    #     """Tests error checking on TaxPlot is sane for the title_text class"""
    #     # Checks title handling is sane.
    #     self.base_test.title_text = ['Foo']
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_use_latex(self):
    #     """Tests error handling with use_latex class"""
    #     self.base_test.use_latex = 'Foo'
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # def test_check_base_latex_family_class(self):
    #     """Chesks error handling with latex_family"""
    #     self.base_test.latex_family = 'foo'
    #     self.assertRaises(ValueError, self.base_test.check_base)

    # def test_check_base_latex_font_class(self):
    #     """Chesks error handling with latex_font class"""
    #     self.base_test.latex_font = 3
    #     self.assertRaises(TypeError, self.base_test.check_base)

    # # Tests check_barchart
    # def test_check_barchart_bar_width_class(self):
    #     """Tests check_barchart handles bar_width class checking sanely"""
    #     self.bar_test.bar_width = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_x_tick_interval_class(self):
    #     """Tests check_barchart handles x_tick_interval class sanely"""
    #     self.bar_test.x_tick_interval = 'foo'
    #     self.assertRaises(TypeError, self.bar_test.check_barchart)

    # def test_check_barchart_width_and_interval(self):
    #     """Tests check_barchart handles a greater width sanely"""
    #     self.bar_test.bar_width = 3
    #     self.bar_test.x_tick_interval = 1
    #     self.assertRaises(ValueError, self.bar_test.check_barchart)

    # def test_check_barchart_x_min_class(self):
    #     """Tests check_barchart handles x_min class sanely"""
    #     self.bar_test.x_min = 'foo'
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

    # # Tests check_piechart
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

    # def test_check_piechart_axis_lims_class(self):
    #     """Tests check_piechart throws an error axis_lims is not iterable"""
    #     self.pie_test.axis_lims = 'foo'
    #     self.assertRaises(TypeError, self.pie_test.check_piechart)

    # def test_check_piechart_axis_lims_length(self):
    #     """Tests check_piechart errors when axis_lim length is wrong."""
    #     self.pie_test.axis_lims = []
    #     self.assertRaises(ValueError, self.pie_test.check_piechart)

    # def test_check_piechart_axis_lims_values(self):
    #     """Tests check_piechart errors when the axis_min is greater"""
    #     self.pie_test.axis_lims = [1, 0.5]
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

    # def test_check_scatter_show_r2_class(self):
    #     """Tests checks_scatter errors when show_r2 is not boolian"""
    #     self.scatter_test.show_r2 = 'foo'
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

    # def test_check_scatter_x_axis_dimensions_class(self):
    #     """Tests check_scatter errors when x_axis_dimensions is not a tuple"""
    #     self.scatter_test.x_axis_dimensions = 'foo'
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_marker_class(self):
    #     """Tests check_scatter errors when marker is not a reasonable class"""
    #     self.scatter_test.markers = 3
    #     self.assertRaises(TypeError, self.scatter_test.check_scatter)

    # def test_check_scatter_marker_iterable_class(self):
    #     """Tests check_scatter errors when iterable markers are not strings"""
    #     self.scatter_test.markers = [1]
    #     self.assertRaises(ValueError, self.scatter_test.check_scatter)

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

    # # Tests set_colormap
    # def test_set_colormap_class(self):
    #     """Checks set_colormap throws an error when the class is wrong"""
    #     # Colormap must be None, a string, or numpy array
    #     self.assertRaises(TypeError, self.base_test.set_colormap, ['Foo'])

    # def test_set_colormap_no_data(self):
    #     """Checks set_colormap does nothing when there is no data"""
    #     # Checks that nothing is returned when data is none.
    #     self.base_test.data = None
    #     self.base_test.colors = 'Spectral'
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      self.colormap).all())

    # def test_set_colormap_none(self):
    #     """Checks set_colormap handles sanely when colormap is None."""
    #     # Checks what happens when data is there and colormap is None.
    #     self.base_test.data = self.data
    #     self.base_test.colors = None
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      self.colormap).all())

    # def test_set_colormap_string(self):
    #     """Checks set_coloramp handles sanely when colormap is a string"""
    #     # Tests colormap assignment when colormap is a string
    #     self.base_test.colors = 'Spectral'
    #     known_colormap = translate_colorbrewer(4, 'Spectral')
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap ==
    #                      known_colormap).all())

    # def test_set_colormap_array(self):
    #     """Checks set_coloramp handles sanely when colormap is an array"""
    #     # Tests colormap assignment when colormap is an array with fewer rows
    #     # than are supplied in data
    #     colors = array([[1, 2, 3], [1, 2, 3]])
    #     self.base_test.colors = colors
    #     self.assertRaises(ValueError, self.base_test.set_colormap)

    #     # Tests colors when a full array has been supplied
    #     colors = array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    #     self.base_test.colors = colors
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__colormap == colors).all())

    # def test_set_colormap_show_edge(self):
    #     """Checks set_colormap can show_edge sanely"""
    #     self.base_test.show_edge = True
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__edgecolor ==
    #                      zeros(self.colormap.shape)).all())

    # def test_set_colormap_hide_edge(self):
    #     self.base_test.show_edge = False
    #     self.base_test.set_colormap()
    #     self.assertTrue((self.base_test._TaxPlot__edgecolor ==
    #                      self.colormap).all())

    # # Tests set_dimensions
    # def test_set_dimensions_return(self):
    #     """Checks set_dimensions works correctly with sane inputs"""
    #      # Sets up fig_dims and axis_dis
    #     self.base_test.fig_dims = (4, 6)
    #     self.base_test.axis_dims = (0.2, 0.2, 0.7, 0.7)
    #     self.base_test.set_dimensions()
    #     axis_bounds = self.base_test._TaxPlot__axes.get_position().bounds
    #     # Checks an error is thrown
    #     assert_array_equal(array(list(self.base_test.fig_dims)),
    #                        self.base_test._TaxPlot__fig.get_size_inches())
    #     assert_almost_equal(self.base_test.axis_dims, axis_bounds, decimal=4)

    # # Tests set_dist_dimensions
    # def test_set_scatter_dimensions_fig_dimensions(self):
    #     """Tests set_scatter_dimensions handles figure_dimensions correctly"""
    #     # Sets up known dimensions
    #     fig_dims = (4, 6)
    #     # Updates the dimensions
    #     self.scatter_test.fig_dims = fig_dims
    #     self.scatter_test.set_scatter_dimensions()
    #     # Checks the figure dimensions are sane
    #     self.assertEqual(self.scatter_test.fig_dims, fig_dims)
    #     assert_array_equal(self.scatter_test._TaxPlot__fig.get_size_inches(),
    #                        array(self.scatter_test.fig_dims))

    def test_set_scatter_dimensions_axes_dims_no_axes_no_show(self):
        """Tests set_scatter_dimensions handles no axis and no show sanely"""
        # Sets up axis dimensions
        axis_dims = (0.2, 0.2, 0.7, 0.7)
        # Sets up ScatterPlot properties
        self.scatter_test.axis_dims = axis_dims
        self.scatter_test._TaxPlot__axes = None
        self.scatter_test.show_distribution = False
        self.scatter_test._ScatterPlot__x_axes = None
        self.scatter_test._ScatterPlot__y_axes = None
        # Updates the axis dimensions
        self.scatter_test.set_scatter_dimensions()
        # Checks axes are created correctly
        self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
        self.assertTrue(self.scatter_test._ScatterPlot__x_axes is None)
        self.assertTrue(self.scatter_test._ScatterPlot__x_axes is None)
        # Checks the axis dimensions are sane
        main_axis_bounds = \
            self.scatter_test._TaxPlot__axes.get_position().bounds
        self.assertEqual(self.scatter_test.axis_dims, axis_dims)
        self.assertEqual(self.scatter_test.axis_dims, main_axis_bounds)

    def test_set_scatter_dimensions_pre_axes_no_show(self):
        """Tests set_scatter_dimensions handles axes and no show sanely"""
        # Checks the default axis dimensions are set for an existing axis
        default_dims = (0.09375, 0.125, 0.625, 0.666667)
        self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
        pre_axis_dims = self.scatter_test._TaxPlot__axes.get_position().bounds
        self.assertEqual(default_dims, pre_axis_dims)
        # Sets new axis dimenions
        axis_dims = (0.2, 0.2, 0.7, 0.7)
        self.scatter_test.axis_dims = axis_dims
        self.scatter_test._TaxPlot__axes = None
        self.scatter_test.show_distribution = False
        self.scatter_test._ScatterPlot__x_axes = None
        self.scatter_test._ScatterPlot__y_axes = None
        # Updates the axis dimensions
        self.scatter_test.set_scatter_dimensions()
        # Checks the axes are created correctly
        self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
        self.assertTrue(self.scatter_test._ScatterPlot__x_axes is None)
        self.assertTrue(self.scatter_test._ScatterPlot__x_axes is None)
        # Checks the axis dimensions are sane
        main_axis_bounds = \
            self.scatter_test._TaxPlot__axes.get_position().bounds
        self.assertEqual(self.scatter_test.axis_dims, axis_dims)
        self.assertEqual(self.scatter_test.axis_dims, main_axis_bounds)
        
        


        # Checks an axis has been created
        # self.assertTrue(isinstance(self.scatter_test._TaxPlot__axes, plt.Axes))
        # Checks the axis is the appropriate dimensions
        # axis_bounds = self.base_test._TaxPlot__axes.get_position().bounds
        # assert_array_equal(array(list(self.base_test.fig_dims)),
        #                    self.base_test._TaxPlot__fig.get_size_inches())
        # assert_almost_equal(self.base_test.axis_dims, axis_bounds, decimal=4)
        # # Checks no axes exist for the x and y axes
        # self.assertEqual(self._ScatterPlot__x_axes, None)
        # self.assertEqual(self._ScatterPlot__y_axes, None)
        # Checks the figure only has one set of axes
        # self.assertEqual(len(self.base_test._TaxPlot__fig.get_axes), 1)

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

    # # Tests the set of get functions
    # def test_get_colormap(self):
    #     """Checks the colormap can be sanely returned"""
    #     known = self.base_test._TaxPlot__colormap
    #     test = self.base_test.get_colormap()
    #     self.assertTrue((known == test).all())

    # def test_get_dimensions(self):
    #     """Checks the figure and axis dimensions can be returned sanely"""
    #     known_fig = self.base_test._TaxPlot__fig_dims
    #     known_axis = self.base_test._TaxPlot__axis_dims
    #     (test_fig, test_axis) = self.base_test.get_dimensions()
    #     self.assertEqual(known_fig, test_fig)
    #     self.assertEqual(known_axis, test_axis)

    # def test_get_filepath_no_filetype(self):
    #     """Checks the filepath can be returned sanely when none was supplied"""
    #     known = None
    #     self.base_test._TaxPlot__filetype
    #     test = self.base_test.get_filepath()
    #     self.assertEqual(known, test)

    # def test_get_filepath_string(self):
    #     """Checks the filepath can be returned sanely when its a string"""
    #     known = '/Users/jwdebelius/Desktop/test.pdf'
    #     self.base_test = self.base_test.set_filepath(known)
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

    # # Tests render_barchart
    # def test_render_barchart_defaults(self):
    #     """Checks the barchart can be rendered sanely"""
    #     # Generates a bar chart using the test data
    #     self.bar_test.render_barchart()
    #     test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     self.bar_test.set_filepath(test_filename)
    #     self.bar_test.save_figure()
    #     # Sets up the known filestring
    #     known_file = open(pjoin(TEST_DIR, 'files/known_bar_default.svg'), 'U')
    #     known_fig = known_file.read()
    #     known_file.close()
    #     # Reads in the test figure
    #     test_file = open(test_filename, 'U')
    #     test_fig = test_file.read()
    #     test_file.close()
    #     self.assertEqual(known_fig, test_fig)
    #     # Removes the test figure
    #     remove(test_filename)

    # def test_render_barchart_no_lables_no_leg(self):
    #     """Checks the barchart rendering when defaults are changed"""
    #     # Generates a bar chart using the test data
    #     self.bar_test.match_legend = False
    #     self.bar_test.show_x_labels = False
    #     self.bar_test.show_y_labels = False
    #     self.bar_test.render_barchart()
    #     test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     self.bar_test.set_filepath(test_filename)
    #     self.bar_test.save_figure()
    #     # Sets up the known filestring
    #     known_file = open(pjoin(TEST_DIR, 'files/known_bar_switch.svg'), 'U')
    #     known_fig = known_file.read()
    #     known_file.close()
    #     # Reads in the test figure
    #     test_file = open(test_filename, 'U')
    #     test_fig = test_file.read()
    #     test_file.close()
    #     self.assertEqual(known_fig, test_fig)
    #     # Removes the test figure
    #     remove(test_filename)

    # # Tests render_legend
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

    # def test_render_legend_with_error(self):
    #     """Checks the barchart can be rendered sanely with errorbars"""
    #     test = BarChart(fig_dims=(6, 3), axis_dims=(0.2, 0.35, 0.4, 0.5),
    #                     data=self.data, groups=self.groups,
    #                     samples=self.samples, error=self.error,
    #                     colormap='Spectral', show_legend=True,
    #                     legend_offset=(1.9, 0.9), show_error=True)
    #     test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     test = test.set_filepath(test_filename)
    #     test = test.render_barchart()
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
    #     self.assertTrue(isinstance(test._TaxPlot__legend, Legend))
    #     self.assertEqual(known_fig, test_fig)
    #     # Removes the test figure
    #     remove(test_filename)

    # # Tests render_title
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

    # def test_render_title(self):
    #     """Test render_title respond sanely"""
    #     # Generates a bar chart using the test data
    #     self.bar_test.title_text = 'HP Villians'
    #     self.bar_test.show_title = True
    #     self.bar_test.render_barchart()
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

    # # Tests render_piechart
    # def test_render_piechart_warning(self):
    #     """Checks a warning is triggered when data is a matrix"""
    #     with catch_warnings(record=True) as w:
    #         # Triggers the warning
    #         test = PieChart(fig_dims=self.fig_dims,
    #                         axis_dims=self.axis_dims,
    #                         data=self.data,
    #                         groups=self.groups,
    #                         samples=self.samples)
    #         test.render_piechart()
    #         # Verifies somethings about the warning
    #         self.assertTrue(len(w) == 1)
    #         self.assertTrue(issubclass(w[-1].category, UserWarning))

    # def test_render_piechart(self):
    #     """Checks the default rendering for the piechart"""
    #     # Generates a bar chart using the test data
    #     # test_filename = pjoin(TEST_DIR, 'files/test.svg')
    #     test_filename = '/Users/jwdebelius/Desktop/test.pdf'
    #     test = PieChart(data=self.data[:, 0],
    #                     samples=[self.samples[0]],
    #                     groups=self.groups,
    #                     colors='RdPu',
    #                     filename=test_filename,
    #                     plot_ccw=False)
    #     print test.axis_dims
    #     print test._TaxPlot__axes
    #     print test._TaxPlot__fig.get_axes()
    #     test.render_piechart()
    #     test.save_figure()
    #     # # Sets up the known filestring
    #     # known_file = open(pjoin(TEST_DIR, 'files/known_piechart.svg'), 'U')
    #     # known_fig = known_file.read()
    #     # known_file.close()
    #     # # Reads in the test figure
    #     # test_file = open(test_filename, 'U')
    #     # test_fig = test_file.read()
    #     # test_file.close()
    #     # self.assertEqual(known_fig, test_fig)
    #     # # Removes the test figure
    #     # remove(test_filename)

    # # Tests save_figure.
    # # Other tests are included in the render_X tests (the figures get saved)
    # def test_save_fig_no_filepath(self):
    #     """Checks that a file is not generated when the filetype is None."""
    #     test_filename = pjoin(TEST_DIR, 'files/test.pdf')
    #     self.base_test.set_filepath(test_filename)
    #     self.base_test._TaxPlot__filetype = None
    #     self.base_test.save_figure()
    #     self.assertFalse(exists(test_filename))


if __name__ == '__main__':
    main()
