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
from TaxPlot import TaxPlot
from ScatterPlot import ScatterPlot
from americangut.make_phyla_plots import translate_colorbrewer

# Determines the location fo the reference files
TEST_DIR = dirname(realpath(__file__))

# Stops warnings from being triggered
filterwarnings("always")


class TestTaxPlot(TestCase):

    def setUp(self):
        """Sets up variables for testing"""
        # Sets up distributions
        self.ind = array([01.000,  2.000,  3.000,  4.000,  5.000,
                          06.000,  7.000,  8.000,  9.000, 10.000,
                          11.000, 12.000, 13.000, 14.000, 15.000,
                          16.000, 17.000, 18.000, 19.000, 20.000,
                          21.000, 22.000, 23.000, 24.000, 25.000])
        self.dep = array([5.667,  5.960,  6.501,  7.273,   7.346,
                          08.075,  8.773,  8.799,  9.543, 09.904,
                          10.634, 10.946, 11.545, 12.114, 12.542,
                          12.927, 13.579, 13.935, 14.486,  14.945,
                          15.300, 15.895, 16.623, 16.940,  17.459])
        self.err = array([00.320,  0.465,  0.674,  0.445,  0.433,
                          00.498,  0.552,  0.344,  0.758,  0.542,
                          00.594,  0.378,  0.390,  0.441,  0.378,
                          00.503,  0.446,  0.445,  0.545,  0.537,
                          00.456,  0.361,  0.562,  0.486,  0.506])
        self.samples = ['Basilesk Venom']
        self.colormap = array([[0.00, 0.00, 0.00]])
        self.scatter_test = ScatterPlot(data=self.dep,
                                        groups=self.ind,
                                        error=self.err,
                                        samples=['Basilesk Venom'])
        self.scatter_properties = set(['data', 'groups', 'samples', 'error',
                                       'show_error', '_TaxPlot__error_bars',
                                       'fig_dims', '_TaxPlot__fig',
                                       'axis_dims', '_TaxPlot__axes',
                                       '_TaxPlot__filepath',
                                       '_TaxPlot__filename',
                                       '_TaxPlot__filetype',
                                       'save_properties', 'show_edge',
                                       'colors', '_TaxPlot__colormap',
                                       '_TaxPlot__edgecolor',
                                       'show_legend', 'legend_offset',
                                       'legend_properties',
                                       '_TaxPlot__patches',
                                       '_TaxPlot__legend', 'show_axes',
                                       'show_frame', 'axis_properties',
                                       'show_title', 'title_text',
                                       'title_properties', '_TaxPlot__title',
                                       'use_latex', 'latex_family',
                                       'latex_font', '_TaxPlot__font_set',
                                       '_TaxPlot__properties',
                                       'show_distribution',
                                       'show_dist_hist',
                                       'show_reg_line', 'match_reg_line',
                                       'show_reg_equation', 'show_error',
                                       'show_r2', 'connect_points',
                                       'equation_position', 'r2_position',
                                       'connect_points', 'x_axis_dims',
                                       'y_axis_dims', 'markers', 'bins',
                                       'round_to_x', 'round_to_y', 'normalize',
                                       'x_range', 'y_range',
                                       '_ScatterPlot__x_axes',
                                       '_ScatterPlot__y_axes',
                                       '_ScatterPlot__x_dist',
                                       '_ScatterPlot__y_dist',
                                       '_ScatterPlot__x_bins',
                                       '_ScatterPlot__y_bins',
                                       '_ScatterPlot__x_reg',
                                       '_ScatterPlot__y_reg',
                                       '_ScatterPlot__reg_eqs', 'priority',
                                       '_ScatterPlot__defaults',
                                       '_TaxPlot__defaults'])

    def tearDown(self):
        """Handles teardown of the test object"""
        plt.close('all')

    # Test intilization of classes
    def test_scatter_init(self):
        """Checks a ScatterChart object initializes sanely"""
        # Sets up the known default values
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
        normalize = True
        x_range = None
        y_range = None
        # Compares the known default properties to the instance properties
        self.assertEqual(self.scatter_test.show_distribution,
                         show_distribution)
        self.assertEqual(self.scatter_test.show_dist_hist,
                         show_dist_hist)
        self.assertEqual(self.scatter_test.show_reg_line, show_reg_line)
        self.assertEqual(self.scatter_test.match_reg_line, match_reg_line)
        self.assertEqual(self.scatter_test.show_reg_equation,
                         show_reg_equation)
        self.assertEqual(self.scatter_test.show_error, show_error)
        self.assertEqual(self.scatter_test.show_r2, show_r2)
        self.assertEqual(self.scatter_test.connect_points, connect_points)
        self.assertEqual(self.scatter_test.normalize, normalize)
        self.assertEqual(self.scatter_test.equation_position,
                         equation_position)
        self.assertEqual(self.scatter_test.r2_position, r2_position)
        self.assertEqual(self.scatter_test.x_axis_dims,
                         x_axis_dims)
        self.assertEqual(self.scatter_test.y_axis_dims,
                         y_axis_dims)
        self.assertEqual(self.scatter_test.markers, markers)
        self.assertEqual(self.scatter_test.bins, bins)
        self.assertEqual(self.scatter_test.round_to_x, round_to_x)
        self.assertEqual(self.scatter_test.round_to_y, round_to_y)
        self.assertEqual(self.scatter_test.x_range, x_range)
        self.assertEqual(self.scatter_test.y_range, y_range)
        self.assertEqual(self.scatter_properties,
                         self.scatter_test._TaxPlot__properties)

    # Tests check_scatter
    def test_check_scatter_show_distribution_class(self):
        """Tests checks_scatter errors when show_distribution is not boolian"""
        self.scatter_test.show_distribution = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_show_dist_hist_class(self):
        """Tests checks_scatter errors when show_dist_hist is not boolian"""
        self.scatter_test.show_dist_hist = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_show_reg_line_class(self):
        """Tests checks_scatter errors when show_reg_line is not boolian"""
        self.scatter_test.show_reg_line = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_match_reg_line_class(self):
        """Tests checks_scatter errors when match_reg_line is not boolian"""
        self.scatter_test.match_reg_line = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_show_reg_equation_class(self):
        """Tests checks_scatter errors when connect_points is not boolian"""
        self.scatter_test.show_reg_equation = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_show_r2_class(self):
        """Tests checks_scatter errors when show_r2 is not boolian"""
        self.scatter_test.show_r2 = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_show_error_class(self):
        """Tests checks_scatter errors when show_error is not boolian"""
        self.scatter_test.show_error = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_normalize_class(self):
        """Tests checks_scatter errors when normalize is not boolian"""
        self.scatter_test.normalize = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_connect_points_class(self):
        """Tests checks_scatter errors when connect_points is not boolian"""
        self.scatter_test.normalize = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_equation_position_class(self):
        """Tests check_scatter error when equation_positions is not a tuple"""
        self.scatter_test.equation_position = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_r2_position_class(self):
        """Tests check_scatter error when equation_positions is not a tuple"""
        self.scatter_test.r2_position = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_x_axis_dims_class(self):
        """Tests check_scatter errors when x_axis_dims is not a tuple"""
        self.scatter_test.x_axis_dims = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_y_axis_dims_class(self):
        """Tests check_scatter errors when y_axis_dims is not a tuple"""
        self.scatter_test.y_axis_dims = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_marker_class(self):
        """Tests check_scatter errors when marker is not a reasonable class"""
        self.scatter_test.markers = 3
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_marker_iterable_class(self):
        """Tests check_scatter errors when iterable markers are not strings"""
        self.scatter_test.markers = [1]
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_bins_class(self):
        """Tests check_scatter errors when bins is not an integer"""
        self.scatter_test.bins = 3.1415
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_bins_value(self):
        """Tests check_scatter errors when bins is less than 1"""
        self.scatter_test.bins = -5
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_round_to_x_class(self):
        """Tests check_scatter errors when round_to is not an integer"""
        self.scatter_test.round_to_x = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_round_to_x_value(self):
        """Tests check_scatter errors when round_to is less than 1"""
        self.scatter_test.round_to_x = -5
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_round_to_y_class(self):
        """Tests check_scatter errors when round_to is not an integer"""
        self.scatter_test.round_to_y = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_round_to_y_value(self):
        """Tests check_scatter errors when round_to is less than 1"""
        self.scatter_test.round_to_y = -5
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_x_range_class(self):
        """Tests check_scatter errors when x_range is not iterable"""
        self.scatter_test.x_range = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_x_range_len(self):
        """Tests checks_scatter errors when the length of x_range is not 2"""
        self.scatter_test.x_range = ['foo']
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_x_range_numeric(self):
        """Tests check_scatter errors when x_range elements are not numbers"""
        self.scatter_test.x_range = ['foo', 'bar']
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_x_range_order(self):
        """Tests check_scatter errors when the first x_range is greater."""
        self.scatter_test.x_range = [1, 0]
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_y_range_class(self):
        """Tests check_scatter errors when y_range is not iterable"""
        self.scatter_test.y_range = 'foo'
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_y_range_len(self):
        """Tests checks_scatter errors when the length of y_range is not 2"""
        self.scatter_test.y_range = ['foo']
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    def test_check_scatter_y_range_numeric(self):
        """Tests check_scatter errors when y_range elements are not numbers"""
        self.scatter_test.y_range = ['foo', 'bar']
        self.assertRaises(TypeError, self.scatter_test.check_scatter)

    def test_check_scatter_y_range_order(self):
        """Tests check_scatter errors when the first y_range is greater."""
        self.scatter_test.y_range = [1, 0]
        self.assertRaises(ValueError, self.scatter_test.check_scatter)

    # Calculates the smoothed range
    def calculate_smoothed_range_custom_x(self):
        """Tests that calculate_smoothed_range ignores custom x_ranges"""
        # Sets a custom range
        known_range = [0, 300]
        self.scatter_test.x_range = known_range
        # Calculates the best range for the data
        self.scatter_test.calculate_smoothed_range()
        # Checks the known matches the test value
        self.assertEqual(known_range, self.scatter_test.x_range)

    def calculate_smoothed_range_custom_y(self):
        """Tests calculate_smoothed_range ignores custom y_ranges"""
        # Sets a custom range
        known_range = [0, 300]
        self.scatter_test.y_range = known_range
        # Calculates the best range for the data
        self.scatter_test.calculate_smoothed_range()
        # Checks the known matches the test value
        self.assertEqual(known_range, self.scatter_test.y_range)

    def calculate_smoothed_range_default_x(self):
        """Tests calculate_smoothed_range can calculate a sane x range"""
        # Sets up the known range for the default x parameters
        known_range = [0, 25]
        self.scatter_test.x_range = None
        # Calculates the best range for hte data
        self.scatter_test.calculate_smoothed_range()
        # Checks the known matches the test value
        self.assertEqual(known_range, self.scatter_test.x_range)

    def calculate_smoothed_range_default_y(self):
        """Tests calculate_smoothed_range can calculate a sane y range"""
        # Sets up the known range for the default x parameters
        known_range = [4, 18]
        self.scatter_test.x_range = None
        self.scatter_test.round_to_y = 2
        # Calculates the best range for hte data
        self.scatter_test.calculate_smoothed_range()
        # Checks the known matches the test value
        self.assertEqual(known_range, self.scatter_test.y_range)

    # Tests calculate_distribution
    def test_calculate_distribution_trace(self):
        """Tests calculate_distribution bins data correctly for trace plots"""
        # Sets up the known (default) values for the distribution
        x_dist = array([0.032,  0.040,  0.040,  0.040,  0.048])
        x_bins = array([2.5,  7.5, 12.5, 17.5, 22.5])
        y_dist = [array([0.06666667, 0.09333333, 0.08000000,
                         0.08000000, 0.01333333])]
        y_bins = [array([6.5,  9.5, 12.5, 15.5, 18.5])]
        # Sets a custom number of bins
        test = ScatterPlot(data=self.dep, groups=self.ind,
                           samples=['[Basilesk Venom]'], bins=5,
                           normalize=True, y_range=[5, 20],
                           show_dist_hist=False)
        # Calculates the distribution
        test.calculate_distribution()
        # Checks the x_axis related data
        assert_almost_equal(test._ScatterPlot__x_dist, x_dist, decimal=5)
        assert_array_equal(test._ScatterPlot__x_bins, x_bins)
        # Checks the y_axis related data
        for idx, a in enumerate(test._ScatterPlot__y_dist):
            b = test._ScatterPlot__y_bins[idx]
            assert_almost_equal(y_dist[idx], a, decimal=5)
            assert_array_equal(y_bins[idx], b)

    def test_calculate_distribution_histogram(self):
        """Tests calculate_distribution bins data correctly for histograms"""
        # Sets up the known (default) values for the distribution
        x_dist = array([4,  5,  5,  5,  6])
        x_bins = array([0,  5, 10, 15, 20])
        y_dist = [array([5,  7,  6,  6,  1])]
        y_bins = [array([5,  8, 11, 14, 17])]
        # Sets a custom number of bins and turns off normalization
        self.scatter_test.bins = 5
        self.scatter_test.normalize = False
        self.scatter_test.show_dist_hist = True
        # Calculates the distribution
        self.scatter_test.calculate_distribution()
        # Checks the x_axis related data
        assert_almost_equal(self.scatter_test._ScatterPlot__x_dist, x_dist,
                            decimal=5)
        assert_array_equal(self.scatter_test._ScatterPlot__x_bins, x_bins)
        # # # Checks the y_axis related data
        for idx, a in enumerate(self.scatter_test._ScatterPlot__y_dist):
            b = self.scatter_test._ScatterPlot__y_bins[idx]
            assert_almost_equal(y_dist[idx], a, decimal=5)
            assert_array_equal(y_bins[idx], b)

    # Tests calculate_regression
    def test_calculate_regression(self):
        """ """
        pass

if __name__ == '__main__':
    main()
