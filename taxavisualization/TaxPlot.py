# from numpy import array
# from os import getcwd
from __future__ import division
from os import mkdir, getcwd
from os.path import splitext, split as fsplit, join as pjoin, exists, isfile
from numpy import arange, zeros, ndarray, array
from matplotlib import use, rc
use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from inspect import getmembers, ismethod
from warnings import warn, filterwarnings
from CatTableKeys import check_data_array
from americangut.make_phyla_plots import translate_colorbrewer

# Sets up warning activity to always
filterwarnings("always")


class TaxPlot:
    """Doc String Here"""
    # Sets up basic figure properties that must be defined for each TaxPlot
    # instance
    data = None
    groups = None
    samples = None
    error = None
    # Handles error-associated properties
    show_error = False
    __error_bars = []
    # Sets up figure and axis properties
    fig_dims = None
    __fig = plt.figure()
    axis_dims = None
    __axes = None
    # Sets up saving information
    __filepath = None
    __filename = None
    __filetype = None
    save_properties = {}
    # Handles the colormap properties
    show_edge = False
    colors = None
    __colormap = None
    __edgecolor = None
    # Handles legend properties
    show_legend = False
    legend_offset = None
    legend_properties = {}
    match_legend = True
    __patches = []
    __legend = None
    # Handles axis properties
    show_axes = True
    show_frame = True
    axis_properties = {}
    # Handles the title
    show_title = False
    title_text = None
    title_properties = {}
    __title = None
    # Handles other appearance properties
    use_latex = False
    latex_family = 'sans-serif'
    latex_font = ['Helvetica', 'Arial']
    __properties = None

    # Handles font-related arguments
    __font_set = {'title': FontProperties(family='sans-serif', size=30),
                  'label': FontProperties(family='sans-serif', size=20),
                  'leg': FontProperties(family='sans-serif', size=15),
                  'tick':  FontProperties(family='sans-serif', size=15),
                  'text': FontProperties(family='sans-serif', size=15)}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        """Initializes an instance of a TaxPlot object"""
        # Assembles the defata information
        self.data = data
        self.groups = groups
        self.samples = samples
        self.error = error

        # Adds keyword arguments
        self.add_attributes(**kwargs)

        # Sets up the filepath
        self.set_filepath(filename)
        # Sets up figure and axis properties
        self.set_dimensions()
        # Sets up the colormap
        self.set_colormap()

        # Checks the structure is good
        self.check_base()

    def add_attributes(self, **kwargs):
        """Adds keyword arguments to the object"""
        # Gets the attributes associated with the object
        if self._TaxPlot__properties is None:
            all_attributes = getmembers(self)
            properties = []
            for (k, v) in all_attributes:
                k_check = k in set(['__doc__', '__init__', '__module__'])
                method_check = ismethod(v)
                if k_check or method_check:
                    continue
                properties.append(k)
            self._TaxPlot__properties = set(properties)

        # Adds the attributes to the class instance
        for k, v in kwargs.iteritems():
            if k in self._TaxPlot__properties:
                setattr(self, k, v)
            else:
                raise ValueError('%s is not a TaxPlot property.' % k)

    def check_base(self):
        """Checks that variables in TaxPlot are sane"""
        # Checks the data array is supported
        self.data = check_data_array(self.data, self.groups, self.samples,
                                     'data', 'groups', 'samples')
        # If the error is present, checks the error is supported
        if self.error is not None:
            self.error = check_data_array(self.error, self.groups,
                                          self.samples, 'error', 'groups',
                                          'samples')

        # Checks the show_error argument is sane
        if not isinstance(self.show_error, bool):
            raise TypeError('show_error must be a bool')

         # Checks the figure dimensions are sane
        if not isinstance(self.fig_dims, tuple):
            raise TypeError('fig_dims must be a two-element tuple of numbers')
        for i in self.fig_dims:
            if not isinstance(i, (int, float)):
                raise TypeError('fig_dims must be a two-element tuple of '
                                'numbers')
        if not len(self.fig_dims) == 2:
            raise ValueError('fig_dims must be a two-element tuple of numbers')

         # Checks the axis dimensions are sane
        if not isinstance(self.axis_dims, tuple):
            raise TypeError('axis_dims must be a 4-element tuple.')
        elif not len(self.axis_dims) == 4:
            raise TypeError('axis_dims must be a 4-element tuple.')

        # Checks the save_properties argument is sane
        if not isinstance(self.save_properties, dict):
            raise TypeError('save_properties must be a dictionary')

        # Checks the show_edge argument is sane
        if not isinstance(self.show_edge, bool):
            raise TypeError('show_edge must be a bool')

        # Checks the show_legend arguement is sane
        if not isinstance(self.show_legend, bool):
            raise TypeError('show_legend must be a bool')

        # Checks the legend offset arguemnt is sane
        if self.legend_offset is not None:
            if not isinstance(self.legend_offset, (tuple)):
                raise TypeError('legend_offset must be a tuple')
            elif len(self.legend_offset) not in set([2, 4]):
                raise ValueError('legend_offset must be a tuple')

        # Checks the legend properties argument is sane
        if not isinstance(self.legend_properties, dict):
            raise TypeError('legend_properties must be a dictionary')

        # Checks the match_legend class
        if not isinstance(self.match_legend, bool):
            raise TypeError('match_legend must be a boolian')

        # Checks the show_axes argument is sane
        if not isinstance(self.show_axes, bool):
            raise TypeError('show_axes must be a boolian')

        # Checks the show_frame argument is sane
        if not isinstance(self.show_frame, bool):
            raise TypeError('show_frame must be boolian')

        # Checks the axis_properties arguments are sane
        if not isinstance(self.axis_properties, dict):
            raise TypeError('axis_properties must be a dict')

        # Checks show_title is sane
        if not isinstance(self.show_title, bool):
            raise TypeError('show_title must be boolian')

        # Checks the title class is sane
        title_text_str = isinstance(self.title_text, str)
        if self.title_text is not None and not title_text_str:
            raise TypeError('title must be a string')

        # Checks latex is sane
        if not isinstance(self.use_latex, bool):
            raise TypeError('use_latex must be boolian')

        # Checks the latex_family is sane
        font_families = set(['serif', 'sans-serif', 'cursive', 'fantasy',
                             'monospace'])
        if self.latex_family not in font_families:
            raise ValueError('%s is not a supported latex font family'
                             % self.latex_family)

        # Checks the latex font class is sane
        if not isinstance(self.latex_font, (str, list)):
            raise TypeError('latex font must be a string, or list of strings')

    def write_object_parameters(self, params_file):
        """Writes the parameters into a text file"""
        # Writes the parameters string
        class_name = self.__class__
        print class_name
        object_attributes = self.getmembers()
        params_string = []
        for prop in self._TaxPlot__properties:
            if prop in set(['data', 'error', 'samples', 'groups',
                            '_TaxPlot__properties']):
                continue
            att = object_attributes[prop]
            params_string.append('%s:%s\t%r' % (class_name, prop, att))
        # Writes the parameters file
        if isfile(params_file):
            warn('%s already exists.\nIt will be overwritten.', UserWarning)
        target = open(params_file, 'w')
        target.truncate()
        target.write('\n'.join(params_string))
        target.close()

    def read_object_parameters(self, params_file):
        """"""
        pass

    def get_colormap(self):
        """Retuns the colormap and edgecolor as numpy arrays"""
        return self._TaxPlot__colormap, self._TaxPlot__edgecolor

    def get_dimensions(self):
        """Returns the figure and axis dimensions"""
        return self._TaxPlot__fig_dims, self._TaxPlot__axis_dims

    def get_filepath(self):
        """Returns the save filepath"""
        if self._TaxPlot__filetype is None:
            return None
        return pjoin(self._TaxPlot__filepath, self._TaxPlot__filename)

    def get_font(self, font_type):
        """Returns the specified font"""
        if not font_type in self._TaxPlot__font_set.keys():
            raise ValueError('%s is not a supported font type' % font_type)
        return self._TaxPlot__font_set[font_type]

    def set_filepath(self, filename):
        """Makes sure the save location information is sane"""
        # Checks the file name is supported
        if filename is None:
            self._TaxPlot__filepath = None
            self._TaxPlot__filename = None
            self._TaxPlot__filetype = None
        elif isinstance(filename, str):
            (self._TaxPlot__filepath, self._TaxPlot__filename) = \
                fsplit(filename)
            if self._TaxPlot__filepath == '':
                self._TaxPlot__filepath = getcwd()
            self._TaxPlot__filetype = \
                splitext(self._TaxPlot__filename)[1].upper().replace('.', '')
        else:
            raise TypeError('filepath must be a string')

    def set_dimensions(self):
        """Updates the axis and figure dimensions for a single axis.

        Please note that set_dist_dimensions is a better function to use when
        plotting data with its corresponding x and y distributions."""
        if isinstance(self._TaxPlot__fig, plt.Figure):
            self._TaxPlot__fig.set_size_inches(self.fig_dims)
        if isinstance(self._TaxPlot__axes, plt.Axes):
            self._TaxPlot__axes.set_position(self.axis_dims)
        else:
            self._TaxPlot__axes = self._TaxPlot__fig.add_axes(self.axis_dims)

        # Checks the appropraite number of axes are shown.
        current_axes = self._TaxPlot__fig.get_axes()
        if current_axes > 1:
            for axis in current_axes:
                if not axis.get_position().bounds == self.axis_dims:
                    axis.set_axis_off()

    def set_colormap(self, **kwargs):
        """Updates the colormap and egde color"""
        # Check colormap is a sane classs
        no_colors = self.colors is None
        if not no_colors and not isinstance(self.colors, (str, ndarray)):
            raise TypeError('Colormap must a colorbrewer map name or '
                            'numpy array.')

        # Adds any attributes that might have been supplied (like data)
        self.add_attributes(**kwargs)

        # Handles the colormap is no data is avaliable
        if self.data is not None:
            # Determines the number of rows in the data array
            num_rows = self.data.shape[0]

            # Sets up the colormap when none is provided (default is a gray
            # scale)
            if no_colors:
                interval = 1/num_rows
                color_range = arange(0, 1, interval)
                colors = zeros((num_rows, 3))
                colors[:, 0] = color_range
                colors[:, 1] = color_range
                colors[:, 2] = color_range
                self._TaxPlot__colormap = colors

            # If a string is provided, the colormap is taken from ColorBrewer
            if isinstance(self.colors, str):
                self._TaxPlot__colormap = translate_colorbrewer(num_rows,
                                                                self.colors)

            # Handles colors if map is supplied
            if isinstance(self.colors, ndarray):
                num_colors = len(self.colors[:, 0])
                if num_colors >= num_rows:
                    self._TaxPlot__colormap = self.colors[:(num_rows), :]
                else:
                    raise ValueError('The colormap cannot be used. \nNot'
                                     ' enough colors have been supplied.')

            if self.show_edge:
                self._TaxPlot__edgecolor = zeros(self._TaxPlot__colormap.shape)
            else:
                self._TaxPlot__edgecolor = self._TaxPlot__colormap

    def set_font(self, font_type, font_object):
        """Sets the font object"""
        # Checks the font_type is sane
        if font_type not in self._TaxPlot__font_set:
            raise ValueError('%s is not a supported font type.' % font_type)

        # Checks the font_object is sane
        if not isinstance(font_object, FontProperties):
            raise TypeError('font_object must be a FontProperties instance')

        # Updates the font object
        self._TaxPlot__font_set[font_type] = font_object

    def render_legend(self, **kwargs):
        """Adds a legend to the figure"""
        # Updates the object attributes
        self.add_attributes(**kwargs)

        # Checks the class is sane
        self.check_base()

        # If show_legend is off or there is no potted data, the legend
        # is not updated.
        if self.show_legend and len(self._TaxPlot__patches) > 0:
            # Adds the legend
            if self.match_legend:
                patches = reversed(self._TaxPlot__patches)
            else:
                patches = self._TaxPlot__patches
            font = self._TaxPlot__font_set['leg']
            self._TaxPlot__legend = \
                self._TaxPlot__axes.legend(patches, self.groups, prop=font,
                                           **self.legend_properties)
            # Sets up the offset
            if self.legend_offset is not None:
                self._TaxPlot__legend.set_bbox_to_anchor(self.legend_offset)

    def render_title(self, **kwargs):
        """Adds a title to the figure"""
        # Updates the object attributes
        self.add_attributes(**kwargs)

        # Checks the class is sane
        self.check_base()

        # A title is not added is there are no axes
        if self._TaxPlot__axes is not None and self.show_title:
            # Checks to see that a title can be added
            title_font = self._TaxPlot__font_set['title']
            self._TaxPlot__axes.set_title(self.title_text,
                                          fontproperties=title_font,
                                          **self.title_properties)

    def save_figure(self):
        """Saves the rendered figure"""
        if self._TaxPlot__filetype is not None:
            # Sets up the filepath for saving
            filename = pjoin(self._TaxPlot__filepath, self._TaxPlot__filename)
            # Creates the file directory if necessary
            if not exists(self._TaxPlot__filepath):
                mkdir(self._TaxPlot__filepath)
            # Saves the file
            if len(self.save_properties) > 0:
                self._TaxPlot__fig.savefig(filename, **self.save_properties)
            else:
                self._TaxPlot__fig.savefig(filename)

    def return_figure(self):
        """Returns the figure object"""
        return self._TaxPlot__fig

    def render_latex(self):
        """Renders the figure using latex text"""
        if self.use_latex:
            rc('text', usetex=True)
            rc('font', **{'family': self.latex_family,
                          self.latex_family: self.latex_font})


class BarChart(TaxPlot):
    """Doc string here"""
    # Sets up barchart specific properties
    bar_width = 0.8
    x_tick_interval = 1.0
    x_min = -0.5
    x_font_angle = 45
    x_font_align = 'right'
    show_x_labels = True
    show_y_labels = True
    __bar_left = None
    __all_faces = []

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
        self.check_base()

        # Checks the bar_width class
        if not isinstance(self.bar_width, (int, float)):
            raise TypeError('bar_width must be a number')
        # Checks the x_tick_interval class
        if not isinstance(self.x_tick_interval, (int, float)):
            raise TypeError('x_tick_interval must be a number')
        # Checks the bar_width is not greater than the x_tick_interval
        if self.bar_width > self.x_tick_interval:
            raise ValueError('The bar_width cannot be greater than the '
                             'x_tick_interval.')

        # Checks the x_min argument
        if not isinstance(self.x_min, (int, float)):
            raise TypeError('x_min must be a number')

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

        # Sets things up to render using LaTex
        self.render_latex()

        # Sets up the x_axis properties
        num_samples = len(self.samples)
        x_ticks = arange(num_samples)
        x_max = self.x_min + num_samples*self.x_tick_interval
        self.axis_properties['xlim'] = [self.x_min, x_max]
        self.axis_properties['xticks'] = x_ticks
        # Updates the left side for the barchart
        self._BarChart__bar_left = x_ticks - self.bar_width/2

        # Plots the data
        for count, category in enumerate(self.data):
            bottom_bar = sum(self.data[0:count, :], 0)
            if self.match_legend:
                count_rev = len(self.groups) - (1 + count)
                facecolor = self._TaxPlot__colormap[count_rev, :]
                edgecolor = self._TaxPlot__edgecolor[count_rev, :]
            else:
                facecolor = self._TaxPlot__colormap[count, :]
                edgecolor = self._TaxPlot__edgecolor[count, :]
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

        # Updates the x-axis properties
        self._TaxPlot__axes.set_xlim(self.axis_properties['xlim'])
        self._TaxPlot__axes.set_xticks(self.axis_properties['xticks'])

        # Matches the plotting axis if necessary
        y_ticks = self._TaxPlot__axes.get_yticks()

        # Sets up the x tick labels
        if self.show_x_labels:
            self.axis_properties['xticklabels'] = self.samples
        else:
            self.axis_properties['xticklabels'] = ['']*num_samples

        # Updates the y tick labels
        if self.show_y_labels:
            self.axis_properties['yticklabels'] = map(str, y_ticks)
        else:
            self.axis_properties['yticklabels'] = ['']*len(y_ticks)

        # Updates the axis to include the appropriate x and y tick labels
        xticklabels = self.axis_properties['xticklabels']
        yticklabels = self.axis_properties['yticklabels']
        tick_font = self._TaxPlot__font_set['tick']
        x_align = self.x_font_align
        self._TaxPlot__axes.set_xticklabels(xticklabels,
                                            fontproperties=tick_font,
                                            rotation=self.x_font_angle,
                                            horizontalalignment=x_align)
        self._TaxPlot__axes.set_yticklabels(yticklabels,
                                            fontproperties=tick_font)

        # Adds a legend
        self.render_legend()

        # Adds a title
        self.render_title()


class PieChart(TaxPlot):
    """Doc String Here"""
    # Sets up pie chart specific properties
    plot_ccw = False
    start_angle = 90
    axis_lims = [-1.1, 1.1]
    show_labels = False
    numeric_labels = False
    label_distance = 1.1
    __labels = None

    def __init__(self, data, groups, samples, filename=None, **kwargs):
        """Initializes a PieChart instance"""
        # Sets up the piechart with a legend.
        self.fig_dims = (5, 3)
        self.axis_dims = (0.06, 0.1, 0.48, 0.8)
        self.show_legend = True
        self.legend_offset = (1.8, 0.75)
        # Initializes the object
        TaxPlot.__init__(self, data, groups, samples, error=None,
                         filename=filename, **kwargs)

        self.check_piechart()

    def check_piechart(self):
        """Checks the PieChart parameters are sane"""
        # Preforms an initial check of the data
        self.check_base()

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

        # Checks the sanity of the axis_lims class
        if not isinstance(self.axis_lims, (list, tuple)):
            raise TypeError('axis_lims must be an iterable class')
        if not len(self.axis_lims) == 2:
            raise ValueError('axis_lims must be a two-element iterable class.')
        if self.axis_lims[1] < self.axis_lims[0]:
            raise ValueError('The first axis lim must be smaller than the '
                             'second.')

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
            self._TaxPlot__axes.set_xlim([self.axis_lims[1],
                                          self.axis_lims[0]])
        else:
            self._TaxPlot__axes.set_xlim(self.axis_lims)
        self._TaxPlot__axes.set_ylim(self.axis_lims)
        self._TaxPlot__axes.set_visible(self.show_axes)

        # Adds a figure legend and a title.
        self.render_legend()
        self.render_title()


class ScatterPlot(TaxPlot):
    """Doc string here"""
    # Sets up properties for the distribution axes
    show_distribution = True
    show_dist_hist = False
    show_reg_line = False
    match_reg_line = False
    show_reg_equation = False
    equation_position = ()
    show_r2 = False
    r2_position = ()
    connect_points = False
    x_axis_dimensions = (0.09375, 0.66667, 0.62500, 0.25000)
    y_axis_dimensions = (0.75000, 0.12500, 0.18750, 0.50000)
    markers = ['x', 'o', '.', '^', 's', '*']
    __x_axes = None
    __y_axes = None
    __x_dist = None
    __y_dist = []

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        """Initializes a ScatterPlot instance"""
        # Sets up the axis dimensions and figure dimensions, since the defaults
        # are different for a scatter plot instance
        self.fig_dims = (8, 6)
        self.axis_dims = (0.09375, 0.125, 0.625, 0.666667)

        # Initializes an instance of a ScatterPlot object
        TaxPlot.__init__(self, data, groups, samples, error, filename,
                         **kwargs)
        # Updates the figure dimensions appropriately for the formatted axes.
        # Current axes are removed and updated with the new dimensions
        self._TaxPlot__fig.clf()
        self.set_scatter_dimensions()

    def check_scatter(self):
        """Checks additional properties of the scatter instance are sane"""
        # Checks show distribution
        self.check_base()

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

        # Checks the show_r2 argument
        if not isinstance(self.show_r2, bool):
            raise TypeError('show_r2 must be a bool')

        # Checks the connect_points argument
        if not isinstance(self.connect_points, bool):
            raise TypeError('connect_points must be a bool')

        # Checks the equation_position argument
        if not isinstance(self.equation_position, tuple):
            raise TypeError('equation_position must be a tuple')

        # Checks the r2_position argument
        if not isinstance(self.r2_position, tuple):
            raise TypeError('r2_position must be a tuple')

        # Checks the x_axis_dimensions argument
        if not isinstance(self.x_axis_dimensions, tuple):
            raise TypeError('x_axis_dimensions must be a tuple')

        # Checks the y_axis_dimensions argument
        if not isinstance(self.y_axis_dimensions, tuple):
            raise TypeError('y_axis_dimensions must be a tuple')

        # Checks the marker class
        if not isinstance(self.markers, (str, list, tuple)):
            raise TypeError('markers must be a string or interable class '
                            'of strings.')
        if isinstance(self.markers, (list, tuple)):
            for m in self.markers:
                if not isinstance(m, str):
                    raise ValueError('markers must be a string or interable'
                                     ' class of strings.')

    def set_scatter_dimensions(self):
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
        dist = self.show_distribution
        if dist and isinstance(self._ScatterPlot__x_axes, plt.Axes):
            self._ScatterPlot__x_axes.set_position(self.x_axis_dimensions)
        elif dist:
            self._ScatterPlot__x_axes = \
                self._TaxPlot__fig.add_axes(self.x_axis_dimensions)

        if dist and isinstance(self._ScatterPlot__y_axes, plt.Axes):
            self._ScatterPlot__y_axes.set_position(self.y_axis_dimensions)
        elif dist:
            self._ScatterPlot__y_axes = \
                self._TaxPlot__fig.add_axes(self.y_axis_dimensions)

        # Removes any axes which should not be present
        fig_axes = self._TaxPlot__fig.get_axes()
        for axis in fig_axes:
            # Determines if identified axis is an accepted axis
            axis_bounds = axis.get_position().bounds
            is_main = axis_bounds == self.axis_dims
            is_x_ax = axis_bounds == self.x_axis_dimensions
            is_y_ax = axis_bounds == self.y_axis_dimensions
            if not (is_main and is_x_ax and is_y_ax):
                ax = axis
                ax.set_axis_off()

# class BoxPlot(TaxPlot):
#     """DOC STRING HERE"""
#     # Sets up custom properties for the axes
#     notch = True
#     vertical = True

#     def __init__(self, data, groups, samples, filename=None, **kwargs):
#         """Initializes a ScatterPlot instance"""
#         # Sets up the figure and axis dimesnions for the boxplot
        

    

