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
from os import remove
from os.path import realpath, dirname, join as pjoin, exists
from numpy import array, zeros
from numpy.testing import assert_array_equal, assert_almost_equal
from matplotlib import use
use('agg', warn=False)
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from TaxPlot import TaxPlot, BarChart, PieChart, MetaTrace
from americangut.make_phyla_plots import translate_colorbrewer

# Determines the location fo the reference files
TEST_DIR = dirname(realpath(__file__))


class TestTaxPlot(TestCase):

    def setUp(self):
        """Sets up variables for testing"""
        # Sets up the group distribution
        self.data = array([[0.1, 0.2, 0.3],
                           [0.2, 0.3, 0.4],
                           [0.3, 0.4, 0.1],
                           [0.4, 0.1, 0.2]])
        self.samples = ['Sample 1', 'Sample 2', 'Sample 3']
        self.groups = ['Tax A', 'Tax B', 'Tax C', 'Tax D']
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
                                    samples=['Glu/Fru', 'Glu/Lac'],
                                    colors=array([[1, 0.0, 0],
                                                  [0, 1.0, 0]]))
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
                                    'xlim', 'ylim', 'xticks',
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

    # Test intilization of classes
    def test_base_init(self):
        """Tests the base object initializes correctly"""
        # Sets up known values for default properties
        show_error = False
        show_edge = False
        colors = None
        show_legend = False
        legend_offset = None
        legend_properties = {}
        show_axes = True
        xlim = None
        ylim = None
        xticks = None
        yticks = None
        xticklabels = None
        yticklabels = None
        show_title = False
        title_text = None
        title_properties = {}
        use_latex = False
        latex_family = 'sans-serif'
        latex_font = ['Helvetica', 'Arial']
        # Compares the properties to the known
        self.assertTrue((self.base_test.data == self.data).all())
        self.assertEqual(self.base_test.groups, self.groups)
        self.assertEqual(self.base_test.samples, self.samples)
        self.assertTrue((self.base_test.error == self.error).all())
        self.assertEqual(self.base_test.show_error, show_error)
        self.assertEqual(self.base_test.colors, colors)
        self.assertEqual(self.base_test.show_edge, show_edge)
        self.assertEqual(self.base_test.show_legend, show_legend)
        self.assertEqual(self.base_test.legend_offset, legend_offset)
        self.assertEqual(self.base_test.legend_properties, legend_properties)
        self.assertEqual(self.base_test.show_axes, show_axes)
        self.assertEqual(self.base_test.xlim, xlim)
        self.assertEqual(self.base_test.ylim, ylim)
        self.assertEqual(self.base_test.xticks, xticks)
        self.assertEqual(self.base_test.xticklabels, xticklabels)
        self.assertEqual(self.base_test.yticks, yticks)
        self.assertEqual(self.base_test.yticklabels, yticklabels)
        self.assertEqual(self.base_test.show_title, show_title)
        self.assertEqual(self.base_test.title_text, title_text)
        self.assertEqual(self.base_test.title_properties, title_properties)
        self.assertEqual(self.base_test.use_latex, use_latex)
        self.assertEqual(self.base_test.latex_family, latex_family)
        self.assertEqual(self.base_test.latex_font, latex_font)
        self.assertEqual(self.base_properties,
                         self.base_test._TaxPlot__properties)

    def test_bar_init(self):
        """Test BarChart objects are initialized correctly"""
         # Sets up known values for default properties
        bar_width = 0.8
        xmin = -0.5
        xtick_interval = 1.0
        ytick_interval = 0.2
        x_font_angle = 45
        x_font_align = 'right'
        # Compares the properties to the known
        self.assertEqual(self.bar_test.bar_width, bar_width)
        self.assertEqual(self.bar_test.xmin, xmin)
        self.assertEqual(self.bar_test.xtick_interval, xtick_interval)
        self.assertEqual(self.bar_test.ytick_interval, ytick_interval)
        self.assertEqual(self.bar_test.x_font_angle, x_font_angle)
        self.assertEqual(self.bar_test.x_font_align, x_font_align)
        self.assertEqual(self.bar_properties,
                         self.bar_test._TaxPlot__properties)

    def test_pie_init(self):
        """Checks PieChart Object initialize correctly"""
        # Sets up known default values
        plot_ccw = False
        start_angle = 90
        show_labels = False
        numeric_labels = False
        label_distance = 1.1
        # Compares the properties to the known
        self.assertEqual(self.pie_test.plot_ccw, plot_ccw)
        self.assertEqual(self.pie_test.start_angle, start_angle)
        self.assertEqual(self.pie_test.show_labels, show_labels)
        self.assertEqual(self.pie_test.numeric_labels, numeric_labels)
        self.assertEqual(self.pie_test.label_distance, label_distance)
        self.assertEqual(self.pie_properties,
                         self.pie_test._TaxPlot__properties)

    def test_trace_init(self):
        """Checks a MetaTrace object initialzies correctly"""
        linestyle = 'None'
        marker = 'x'
        self.assertEqual(self.trace_test.linestyle, linestyle)
        self.assertEqual(self.trace_test.marker, marker)

    def test_add_attributes(self):
        """Checks that keyword argument attributes can be added correctly"""
        show_legend = True
        self.base_test.add_attributes(show_legend=show_legend)
        self.assertEqual(self.base_test.show_legend, show_legend)

    # Tests checkbase
    def test_checkbase_show_error_class(self):
        """Tests checkbase throws an error when show_error is not a bool"""
        self.base_test.show_error = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_save_properties_class(self):
        """"Checks the save_properties class handling is sane"""
        self.base_test.save_properties = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_fig_classs(self):
        """Checks that an error is called when fig_dims class is wrong"""
        # Sets up fig_dims and axis_dims
        self.base_test.fig_dims = 'foo'
        # Checks an error is thrown
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_fig_dim_classs(self):
        """Checks that an error is called when fig_dims class is wrong"""
        # Sets up fig_dims and axis_dims
        self.base_test.fig_dims = ('foo')
        # Checks an error is thrown
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_num_fig_dims(self):
        """Checks that an error is called when fig_dims length is wrong"""
        # Sets up fig_dims and axis_dims
        self.base_test.fig_dims = (1, 2, 3)
        # Checks an error is thrown
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_axis_class(self):
        """Checks an error is called when axis_dims is of the wrong class"""
        # Sets up fig_dims and axis_dims
        self.base_test.axis_dims = 'foo'
        # Checks an error is thrown
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_axis_dims_error(self):
        """Checks an error is called when axis_dims is of the wrong class"""
        # Sets up fig_dims and axis_dims
        self.base_test.axis_dims = (1, 2, 3)
        # Checks an error is thrown
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_show_edge_class(self):
        """Checks the show_edge class checking is sane"""
        self.base_test.show_edge = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_show_legend_class(self):
        """Checks the show_edge class checking is sane"""
        self.base_test.show_legend = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_legend_properties(self):
        """Checks sanity of legend_properties class check"""
        # Checks the show_title handling is sane
        self.base_test.legend_properties = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_legend_offset(self):
        """Checks sanity of legend_offset checking"""
        # Checks the show_title handling is sane
        self.base_test.legend_offset = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)
        self.base_test.legend_offset = ('Foo', 'Bar', 'Cat')
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_show_axes(self):
        """Checks sanity of show_axes class check"""
        # Checks the show_title handling is sane
        self.base_test.show_axes = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xlim_class(self):
        """Tests checkbase errors when xlim is not a list or None value"""
        self.base_test.xlim = 'Harry'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xlim_length(self):
        """Tests checkbase errors when xlim does not have two elements"""
        self.base_test.xlim = [0]
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_xlim_numeric(self):
        """Tests checkbase errors when xlim is not a list of numerics"""
        self.base_test.xlim = [0, 'foo']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_ylim_class(self):
        """Tests checkbase errors when xlim is not a list or None value"""
        self.base_test.ylim = 'Harry'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_ylim_length(self):
        """Tests checkbase errors when xlim does not have two elements"""
        self.base_test.ylim = [0]
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_ylim_numeric(self):
        """Tests checkbase errors when xlim is not a list of numerics"""
        self.base_test.ylim = [0, 'foo']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xtick_class(self):
        """Tests checkbase error swhen xtick is not a list"""
        self.base_test.xticks = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xtick_list_of_numbers(self):
        """Tests checkbase error swhen xtick is not a list of numbers"""
        self.base_test.xticks = ['foo', 'bar']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xticklabels_class(self):
        """Tests checkbase errors when xticklabels is not a list"""
        self.base_test.xticklabels = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_xticklabels_list_of_strings(self):
        """Tests checkbase errors when xticklabels is not a list of strings"""
        self.base_test.xticklabels = [1, 2, 3]
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_no_xticks_and_xticklabels(self):
        """Tests checkbase errors when xticklabels is sane but no xticks"""
        self.base_test.xticks = None
        self.base_test.xticklabels = ['foo', 'bar']
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_diff_length_xticks_and_xticklabels(self):
        """Tets checkbase errors when xtick and xticklabels are diff length"""
        self.base_test.xticks = [1, 2, 3]
        self.base_test.xticklabels = ['foo', 'bar']
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_ytick_class(self):
        """Tests checkbase error swhen ytick is not a list"""
        self.base_test.yticks = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_ytick_list_of_numbers(self):
        """Tests checkbase error swhen ytick is not a list of numbers"""
        self.base_test.yticks = ['foo', 'bar']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_yticklabels_class(self):
        """Tests checkbase errors when yticklabels is not a list"""
        self.base_test.yticklabels = 'foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_yticklabels_list_of_strings(self):
        """Tests checkbase errors when yticklabels is not a list of strings"""
        self.base_test.yticklabels = [1, 2, 3]
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_no_yticks_and_yticklabels(self):
        """Tests checkbase errors when yticklabels is sane but no yticks"""
        self.base_test.yticks = None
        self.base_test.yticklabels = ['foo', 'bar']
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_diff_length_yticks_and_yticklabels(self):
        """Tets checkbase errors when ytick and yticklabels are diff length"""
        self.base_test.yticks = [1, 2, 3]
        self.base_test.yticklabels = ['foo', 'bar']
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_xlabel_class(self):
        """Tests checkbase errors when xlabel is not a string"""
        self.base_test.xlabel = ['foo']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_ylabel_class(self):
        """Tests checkbase errors when ylabel is not a string"""
        self.base_test.ylabel = ['foo']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_show_title(self):
        """Tests error checking on TaxPlot is sane for the show_title class"""
        # Checks the show_title handling is sane
        self.base_test.show_title = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_title_text(self):
        """Tests error checking on TaxPlot is sane for the title_text class"""
        # Checks title handling is sane.
        self.base_test.title_text = ['Foo']
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_use_latex(self):
        """Tests error handling with use_latex class"""
        self.base_test.use_latex = 'Foo'
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_latex_family_class(self):
        """Chesks error handling with latex_family"""
        self.base_test.latex_family = 'foo'
        self.assertRaises(ValueError, self.base_test.checkbase)

    def test_checkbase_latex_font_class(self):
        """Chesks error handling with latex_font class"""
        self.base_test.latex_font = 3
        self.assertRaises(TypeError, self.base_test.checkbase)

    def test_checkbase_priority(self):
        """Tests that an error occurs whe priority is not a supported value"""
        self.base_test.priority = 'foo'
        self.assertRaises(ValueError, self.base_test.checkbase)

    # Tests check_bar
    def test_check_barchart_bar_width_class(self):
        """Tests check_barchart handles bar_width class checking sanely"""
        self.bar_test.bar_width = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)

    def test_check_barchart_xtick_interval_class(self):
        """Tests check_barchart handles xtick_interval class sanely"""
        self.bar_test.xtick_interval = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)

    def test_check_barchart_width_and_interval(self):
        """Tests check_barchart handles a greater width sanely"""
        self.bar_test.bar_width = 3
        self.bar_test.xtick_interval = 1
        self.assertRaises(ValueError, self.bar_test.check_barchart)

    def test_check_barchart_xmin_class(self):
        """Tests check_barchart handles xmin class sanely"""
        self.bar_test.xmin = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)

    def test_check_barchart_x_font_angle(self):
        """Tests check_barchart handles x_font_angle sanely"""
        self.bar_test.x_font_angle = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)
        self.bar_test.x_font_angle = -25
        self.assertRaises(ValueError, self.bar_test.check_barchart)

    def test_check_barchart_x_font_align(self):
        """Tests check_barchart handles x_font_align sanely"""
        self.bar_test.x_font_align = 'foo'
        self.assertRaises(ValueError, self.bar_test.check_barchart)

    def test_check_barchart_show_x_labels_class(self):
        """Checks that check_barchart handles the show_x_labels class sanely"""
        self.bar_test.show_x_labels = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)

    def test_check_barchart_show_y_labels_class(self):
        """Checks that check_barchart handles the show_y_labels class sanely"""
        self.bar_test.show_y_labels = 'foo'
        self.assertRaises(TypeError, self.bar_test.check_barchart)

       # Tests check_piechart
    def test_check_piechart_plot_ccw_class(self):
        """Tests checks_piechart handles plot_ccw class sanely"""
        self.pie_test.plot_ccw = 'foo'
        self.assertRaises(TypeError, self.pie_test.check_piechart)

    def test_check_piechart_start_angle_class(self):
        """Tests checks_piechart handles start_angle class sanely"""
        self.pie_test.start_angle = 'foo'
        self.assertRaises(TypeError, self.pie_test.check_piechart)

    def test_check_piechart_start_angle_value(self):
        """Tests checks_piechart handles start_angle constraints sanely"""
        self.pie_test.start_angle = -1
        self.assertRaises(ValueError, self.pie_test.check_piechart)

    def test_check_piechart_show_labels_class(self):
        """Tests checks_piechart handles show_labels class sanely"""
        self.pie_test.show_labels = 'foo'
        self.assertRaises(TypeError, self.pie_test.check_piechart)

    def test_check_piechart_numeric_labels_class(self):
        """Tests checks_piechart handles numeric_labels class sanely"""
        self.pie_test.numeric_labels = 'foo'
        self.assertRaises(TypeError, self.pie_test.check_piechart)

    def test_check_piechart_labels_distance_class(self):
        """Tests checks_piechart errors when label_distance is not numeric"""
        self.pie_test.label_distance = 'foo'
        self.assertRaises(TypeError, self.pie_test.check_piechart)

    # Tests check_trace
    def test_check_trace_linestyle(self):
        """Tests check_trace errors when linestyle is not supported"""
        self.trace_test.linestyle = 'foo'
        self.assertRaises(ValueError, self.trace_test.check_trace)

    def test_check_trace_markers_class(self):
        """Tests check_trace errors when markers is not supported"""
        self.trace_test.marker = 'foo'
        self.assertRaises(ValueError, self.trace_test.check_trace)

    # Tests write_object_paremeters
    def test_write_parameters(self):
        """Tests an object parameters file can be generated sanely"""
        known_params_str = ["TaxPlot:show_title\tFalse\nTaxPlot:axis_dims\t"
                            "(0.3, 0.3, 0.6, 0.6)\nTaxPlot:colors\tNone\n"
                            "TaxPlot:fig_dims\t(3, 4)\nTaxPlot:latex_family"
                            "\tsans-serif\nTaxPlot:latex_font\t['Helvetica',"
                            " 'Arial']\nTaxPlot:legend_offset\tNone\nTaxPlot:"
                            "legend_properties\t{}\nTaxPlot:priority\tCURRENT"
                            "\nTaxPlot:save_properties\t{}\nTaxPlot:show_axes"
                            "\tTrue\nTaxPlot:show_edge\tFalse\nTaxPlot:"
                            "show_error\tFalse\nTaxPlot:show_legend\tFalse\n"
                            "TaxPlot:title_properties\t{}\nTaxPlot:title_text"
                            "\tNone\nTaxPlot:use_latex\tFalse\nTaxPlot:xlabel"
                            "\t\nTaxPlot:xlim\tNone\nTaxPlot:xticklabels\t"
                            "None\nTaxPlot:xticks\tNone\nTaxPlot:ylabel\t\n"
                            "TaxPlot:ylim\tNone\nTaxPlot:yticklabels\tNone\n"
                            "TaxPlot:yticks\tNone"]
        test_string = self.base_test.write_parameters()
        self.assertEqual(known_params_str[0], test_string)

    def test_read_parameters_current_priority(self):
        """Tests an object parameters file can sanely keep current"""
        # Sets the priority on current custom values
        self.priority = 'CURRENT'
        # Loads the parameter file
        self.base_test.read_parameters(self.params_file)
        # Checks the paramaters have been set appropriate
        known_new_colors = 'Spectral'
        known_set_fig_dims = (3, 4)
        self.assertEqual(self.base_test.colors, known_new_colors)
        self.assertEqual(self.base_test.fig_dims, known_set_fig_dims)

    def test_read_parameters_imported_priority(self):
        """Tests read_parameters can sanely use imported values"""
        # Sets the priority to imported values
        self.base_test.priority = 'IMPORTED'
        # Loads the parameter file
        self.base_test.read_parameters(self.params_file)
        # Checks the paramaters have been set appropriate
        known_new_colors = 'Spectral'
        known_set_fig_dims = (6, 4)
        self.assertEqual(self.base_test.colors, known_new_colors)
        self.assertEqual(self.base_test.fig_dims, known_set_fig_dims)

    # Tests update_colormap
    def test_update_colormap_class(self):
        """Checks update_colormap throws an error when the class is wrong"""
        # Colormap must be None, a string, or numpy array
        self.assertRaises(TypeError, self.base_test.update_colormap, ['Foo'])

    def test_update_colormap_no_data(self):
        """Checks update_colormap does nothing when there is no data"""
        # Checks that nothing is returned when data is none.
        self.base_test.data = None
        self.base_test.colors = 'Spectral'
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__colormap ==
                         self.colormap).all())

    def test_update_colormap_none(self):
        """Checks update_colormap handles sanely when colormap is None."""
        # Checks what happens when data is there and colormap is None.
        self.base_test.data = self.data
        self.base_test.colors = None
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__colormap ==
                         self.colormap).all())

    def test_update_colormap_string(self):
        """Checks set_coloramp handles sanely when colormap is a string"""
        # Tests colormap assignment when colormap is a string
        self.base_test.colors = 'Spectral'
        known_colormap = translate_colorbrewer(4, 'Spectral')
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__colormap ==
                         known_colormap).all())

    def test_update_colormap_array(self):
        """Checks set_coloramp handles sanely when colormap is an array"""
        # Tests colormap assignment when colormap is an array with fewer rows
        # than are supplied in data
        colors = array([[1, 2, 3], [1, 2, 3]])
        self.base_test.colors = colors
        self.assertRaises(ValueError, self.base_test.update_colormap)

        # Tests colors when a full array has been supplied
        colors = array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        self.base_test.colors = colors
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__colormap == colors).all())

    def test_update_colormap_show_edge(self):
        """Checks update_colormap can show_edge sanely"""
        self.base_test.show_edge = True
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__edgecolor ==
                         zeros(self.colormap.shape)).all())

    def test_update_colormap_hide_edge(self):
        self.base_test.show_edge = False
        self.base_test.update_colormap()
        self.assertTrue((self.base_test._TaxPlot__edgecolor ==
                         self.colormap).all())

    # Tests update_dimensions
    def test_update_dimensions(self):
        """Checks update_dimensions works correctly with sane inputs"""
         # Sets up fig_dims and axis_dis
        self.base_test.fig_dims = (4, 6)
        self.base_test.axis_dims = (0.2, 0.2, 0.7, 0.7)
        self.base_test.update_dimensions()
        axis_bounds = self.base_test._TaxPlot__axes.get_position().bounds
        # Checks the figures dimensions are sane
        assert_array_equal(array(list(self.base_test.fig_dims)),
                           self.base_test._TaxPlot__fig.get_size_inches())
        assert_almost_equal(self.base_test.axis_dims, axis_bounds, decimal=4)

    # Tests set_filepath
    def test_set_filepath_no_filepath(self):
        """Checks TaxPlot can sanely update the filepath when None is given"""
        # Tries for an instance where filepath is none
        filename = None
        known_filename = None
        known_filepath = None
        known_filetype = None
        self.base_test.set_filepath(filename)
        self.assertEqual(self.base_test._TaxPlot__filepath, known_filepath)
        self.assertEqual(self.base_test._TaxPlot__filename, known_filename)
        self.assertEqual(self.base_test._TaxPlot__filetype, known_filetype)

    def test_set_filepath_string(self):
        """Checks TaxPlot can sanely update a string filepath"""
        # Tries for an instance where filepath is a string
        filename = '$HOME/test/test.txt'
        known_filepath = '$HOME/test'
        known_filename = 'test.txt'
        known_filetype = 'TXT'
        self.base_test.set_filepath(filename)
        self.assertEqual(self.base_test._TaxPlot__filepath, known_filepath)
        self.assertEqual(self.base_test._TaxPlot__filename, known_filename)
        self.assertEqual(self.base_test._TaxPlot__filetype, known_filetype)
        # Tries an instance where the filepath is not a string
        self.assertRaises(TypeError, self.base_test.set_filepath, 3)

    # Tests set_font
    def test_set_font_unsupported_type(self):
        """Checks an error is thrown when the font type is not sane"""
        font_type = 'Foo'
        font_object = FontProperties()
        self.assertRaises(ValueError, self.base_test.set_font, font_type,
                          font_object)

    def test_set_font_unsupported_object(self):
        """Checks an error is thrown when the font object is not sane"""
        font_type = 'title'
        font_object = 'Helvetica'
        self.assertRaises(TypeError, self.base_test.set_font, font_type,
                          font_object)

    def test_set_font(self):
        """Checks that a font can be set sanely using set_font"""
        # Sets up the test values
        font_type = 'title'
        font_object = FontProperties(family='cursive', size=25)
        self.base_test.set_font(font_type, font_object)
        # Checks the known font and the set font are the same
        test = self.base_test._TaxPlot__font_set[font_type]
        self.assertTrue(isinstance(test, FontProperties))
        self.assertEqual(test.get_family(), font_object.get_family())
        self.assertEqual(test.get_name(), font_object.get_name())
        self.assertEqual(test.get_size(), font_object.get_size())
        self.assertEqual(test.get_slant(), font_object.get_slant())
        self.assertEqual(test.get_stretch(), font_object.get_stretch())
        self.assertEqual(test.get_style(), font_object.get_style())
        self.assertEqual(test.get_variant(), font_object.get_variant())
        self.assertEqual(test.get_weight(), font_object.get_weight())

    # Tests the set of "get" functions
    def test_get_colormap(self):
        """Checks the colormap can be sanely returned"""
        known = self.base_test._TaxPlot__colormap
        test = self.base_test.get_colormap()
        self.assertTrue((known == test).all())

    def test_get_filepath_no_filetype(self):
        """Checks the filepath can be returned sanely when none was supplied"""
        known = None
        self.base_test._TaxPlot__filetype
        test = self.base_test.get_filepath()
        self.assertEqual(known, test)

    def test_get_filepath_string(self):
        """Checks the filepath can be returned sanely when its a string"""
        known = '/Users/jwdebelius/Desktop/test.pdf'
        self.base_test.set_filepath(known)
        test = self.base_test.get_filepath()
        self.assertEqual(test, known)

    def test_get_font_unsupported(self):
        """Checks get_font raises an error when the fonttype is unsupported"""
        self.assertRaises(ValueError, self.base_test.get_font, 'foo')

    def test_get_font_supported(self):
        """Checks get_font returns the correct font when type is specified."""
        font = 'title'
        known = self.base_test._TaxPlot__font_set[font]
        test = self.base_test.get_font(font)
        self.assertTrue(isinstance(test, FontProperties))
        self.assertEqual(test.get_family(), known.get_family())
        self.assertEqual(test.get_name(), known.get_name())
        self.assertEqual(test.get_size(), known.get_size())
        self.assertEqual(test.get_slant(), known.get_slant())
        self.assertEqual(test.get_stretch(), known.get_stretch())
        self.assertEqual(test.get_style(), known.get_style())
        self.assertEqual(test.get_variant(), known.get_variant())
        self.assertEqual(test.get_weight(), known.get_weight())

    # Tests save_figure.
    # Other tests are included in the render_X tests (the figures get saved)
    def test_save_fig_no_filepath(self):
        """Checks that a file is not generated when the filetype is None."""
        test_filename = pjoin(TEST_DIR, 'files/file_does_not_exits.file')
        self.base_test.set_filepath(test_filename)
        self.base_test._TaxPlot__filetype = None
        self.base_test.save_figure()
        self.assertFalse(exists(test_filename))

    def test_save_figure_filepath(self):
        """Checks that a file is generated adn the file contains axes"""
        test_filename = pjoin(TEST_DIR, 'files/save_test.pdf')
        # self.assertFalse(exists(test_filename))
        self.base_test.set_filepath(test_filename)
        self.base_test.save_figure()
        self.assertTrue(exists(test_filename))
        remove(test_filename)

    # Tests rendering which does not result in figures
    def test_render_legend_show_false(self):
        """Tests render_legend handles sanely when show_legend is false"""
        # Renders the bar chart so patches are there
        self.bar_test.show_legend = False
        self.bar_test.render_barchart()
        # Sets up the known value of legend
        known_legend = None
        # Tests that no legend is created
        self.bar_test.render_legend()
        self.assertEqual(known_legend, self.bar_test._TaxPlot__legend)

    def test_render_legend_no_patches(self):
        """Tests render_legend handles sanely when no data has been plotted"""
        # Renders the bar chart so patches are there
        self.bar_test.show_legend = True
        self.bar_test._TaxPlot__patches = []
        # Sets up the known value of legend
        known_legend = None
        # Tests that no legend is created
        self.bar_test.render_legend()
        self.assertEqual(known_legend, self.bar_test._TaxPlot__legend)

    def test_render_title_no_axes(self):
        """Checks render_title handles sanely with no axes"""
        self.base_test.show_title = True
        self.base_test._TaxPlot__axes = None
        known_title = None
        self.base_test.render_title()
        self.assertEqual(known_title, self.base_test._TaxPlot__title)

    def test_render_title_show_false(self):
        """Checks render_title handles sanely when show_title is false"""
        self.base_test.show_title = False
        known_title = None
        self.base_test.render_title()
        self.assertEqual(known_title, self.base_test._TaxPlot__title)
 
if __name__ == '__main__':
    main()
