# #!/usr/bin/env python
"""Generates figures from different taxonomic data. Currently, the module can
plot categorical independent data as a bar or pie chart. Continous data can
be plotted as a trace."""

from __future__ import division
from os import mkdir, getcwd
from os.path import splitext, split as fsplit, join as pjoin, exists
from numpy import arange, zeros, ndarray, array
from matplotlib import use, rc
use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from inspect import getmembers, ismethod
from CatTableKeys import check_data_array
from americangut.make_phyla_plots import translate_colorbrewer

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"


class TaxPlot:
    """Base class for making taxa plots

    Parameters
    -----------
    data : ndarray
        The data for the object. Can be a numpy array or numpy vector. Each
        column in data must correspond to a sample (or if its a vector, must
        belong to the same sample). Each row must correspond to a value in
        groups.
    groups : list or 1d ndarray
        The identy of the rows in the array. For bar and pie charts, these
        should be the observation identities, i.e. taxonomy. For traces, these
        are the independent variable associated with the observation.
    samples : list
        Identifies each column in the array.
    error : ndarray, optional
    filename: String, optional
        Directory where output files should be saved. If filename is not
        specified or set, the figure will not be saved.

    Other Parameters
    ----------------
    show_error: bool, optional
        If error has been specified, error bars will be displayed when True.
    fig_dims : tuple
        The size of the figure in inches
    axis_dims : tuple
        Tuple specifying (left, bottom, width, height) of the axes as a
        fraction of the figure dimensions. A value of (0, 0, 1, 1) will take
        up the entire figure.
    save_properties : dict
        A dictionary specifying the way the figure should be saved. Examples
        include specifying dpi or paper-size. See matplotlib savefig for
        properties.
    show_legend: bool
        When true, a legend will be added to the figure
    legend_offset : tuple, optional
        Specifies where the legend should be placed in respect to the axis.
    legend_properties : dict
        Gives any keyword argument associated with legend properties which
        user wishes to specify.
    show_axes : bool
    xlim : list (optional)
    ylim : list (optional)
    xticks : array-like (optional)
    yticks : array_like (optional)
    xticklabels : list (optional)
    yticklabels : list (optional)
    xlabel : str (optional)
    ylabel : str (optional)
    show_title : bool
    title_text : str
    title_properties : dict
    use_latex : bool
    latex_family : str (optional) {'serif', 'sans-serif', 'cursive', 'fantasy',
        'monospace'}
    latex_font : list or string (optional)
    priority : {IMPORTED, CURRENT}
        When a parameter file and the current object have conflicting non-
        defualt values, which setting should be give priority.

    See Also
    --------
    BarChart, PieChart, MetaTrace, matplotlib.pyplot, CatTableKeys

    """
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
    fig_dims = (3, 3)
    __fig = plt.figure(figsize=(3, 3))
    axis_dims = (0.1, 0.1, 0.8, 0.8)
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
    __patches = []
    __legend = None
    # Handles axis properties
    show_axes = True
    xlim = None
    ylim = None
    xticks = None
    yticks = None
    xticklabels = None
    yticklabels = None
    xlabel = ''
    ylabel = ''
    # Handles the title
    show_title = False
    title_text = None
    title_properties = {}
    __title = None
    # Handles other appearance properties
    use_latex = False
    latex_family = 'sans-serif'
    latex_font = ['Helvetica', 'Arial']
    priority = 'CURRENT'
    __properties = None
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
                  'xlim': None,
                  'ylim': None,
                  'xticks': None,
                  'yticks': None,
                  'xticklabels': None,
                  'yticklabels': None,
                  'xlabel': '',
                  'ylabel': '',
                  'show_title': False,
                  'title_text': None,
                  'title_properties': {},
                  'use_latex': False,
                  'latex_family': 'sans-serif',
                  'latex_font': ['Helvetica', 'Arial']}

    # Handles font-related arguments
    __font_set = {'title': FontProperties(size=25),
                  'label': FontProperties(size=20),
                  'leg': FontProperties(size=15),
                  'tick':  FontProperties(size=15),
                  'text': FontProperties(size=15)}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        """Initializes an instance of a TaxPlot object

         Parameters
        -----------
        data : ndarray
            The data for the object. Can be a numpy array or numpy vector. Each
            column in data must correspond to a sample (or if its a vector,
            must belong to the same sample). Each row must correspond to a
            value in groups.
        groups : list or 1d ndarray
            The identy of the rows in the array. For bar and pie charts, these
            should be the observation identities, i.e. taxonomy. For traces,
            these are the independent variable associated with the observation.
        samples : list
            Identifies each column in the array.
        error : ndarray, optional
        filename: String, optional
            Directory where output files should be saved. If filename is not
            specified or set, the figure will not be saved.

        See full doc string for other parameters
        """

        # Assembles the defata information
        self.data = data
        self.groups = groups
        self.samples = samples
        self.error = error

        self._TaxPlot__axes = self._TaxPlot__fig.add_axes(self.axis_dims)

        # Adds keyword arguments
        self.add_attributes(**kwargs)

        # Sets up the filepath
        self.set_filepath(filename)
        # Sets up figure and axis properties
        self.update_dimensions()
        # Sets up the colormap
        self.update_colormap()

        # Checks the structure is good
        self.checkbase()

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

    def checkbase(self):
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

        # Checks the show_axes argument is sane
        if not isinstance(self.show_axes, bool):
            raise TypeError('show_axes must be a boolian')

        # Checks the xlim argument are sane
        if self.xlim is not None:
            if not isinstance(self.xlim, list):
                raise TypeError('xlim must be a list')

            if not len(self.xlim) == 2:
                raise ValueError('xlim must be a two element list')

            xmin_num = isinstance(self.xlim[0], (int, float))
            xmax_num = isinstance(self.xlim[1], (int, float))
            if not (xmin_num and xmax_num):
                raise TypeError('xlim must be a list of numbers.')

        # Checks the ylim argument is sane
        if self.ylim is not None:
            if not isinstance(self.ylim, list):
                raise TypeError('ylim must be a list')

            if not len(self.ylim) == 2:
                raise ValueError('ylim must be a two element list')

            ymin_num = isinstance(self.ylim[0], (int, float))
            ymax_num = isinstance(self.ylim[1], (int, float))
            if not (ymin_num and ymax_num):
                raise TypeError('ylim must be a list of numbers.')

        # Checks the xtick argument is sane
        xticks = self.xticks is not None
        if xticks:
            if not isinstance(self.xticks, (list, ndarray)):
                raise TypeError('xticks must be a list of numbers')
            for xtick in self.xticks:
                if not isinstance(xtick, (float, int)):
                    raise TypeError('xticks must be a list of numbers')

        # Checks the xticklabel argument is sane
        if self.xticklabels is not None:
            if not isinstance(self.xticklabels, list):
                raise TypeError('xticklabels must be a list of strings')
            for xlabel in self.xticklabels:
                if not isinstance(xlabel, str):
                    raise TypeError('xticklabels must be a list of strings')
            if not xticks or not len(self.xticks) == len(self.xticklabels):
                raise ValueError('Each xticklabel must correspond to a tick.')

        # Checks the ytick argument is sane
        yticks = self.yticks is not None
        if yticks:
            if not isinstance(self.yticks, (list, ndarray)):
                raise TypeError('yticks must be a list of numbers')
            for ytick in self.yticks:
                if not isinstance(ytick, (float, int)):
                    raise TypeError('yticks must be a list of numbers')

        # Checks the yticklabel argument is sane
        if self.yticklabels is not None:
            if not isinstance(self.yticklabels, list):
                raise TypeError('yticklabels must be a list of strings')
            for xlabel in self.yticklabels:
                if not isinstance(xlabel, str):
                    raise TypeError('yticklabels must be a list of strings')
            if not yticks or not len(self.yticks) == len(self.yticklabels):
                raise ValueError('Each yticklabel must correspond to a tick.')

        # Checks the xlabel argument is sane
        if self.xlabel is not None and not isinstance(self.xlabel, str):
            raise TypeError('xlabel must be a string.')

        # Checks the ylabel argument is sane
        if self.ylabel is not None and not isinstance(self.ylabel, str):
            raise TypeError('ylabel must be a string.')

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

        # Checks priority is sane
        if self.priority not in set(['CURRENT', 'IMPORTED']):
            raise ValueError('priority must be "CURRENT" or "IMPORTED.')

    def __get_current_parameters(self):
        """Determines the current state of all modifiable parameters"""
        # Gets all the parameters in the object
        all_attributes = getmembers(self)
        # Determines the parameters to be ignored
        exclude = set(['data', 'error', 'samples', 'groups', 'proprity'])
        important_set = set(self._TaxPlot__properties).difference(exclude)
        # Builds a dictionary of the parameter state
        current = {}
        for (attr, val) in all_attributes:
            if not attr in important_set or '__' in attr:
                continue
            prop = '%s' % attr
            current[prop] = val

        return current

    def write_parameters(self):
        """Writes the parameters into a string"""
        # Getes the object class
        class_name = self.__class__
        class_name = '%r' % class_name
        classname = class_name.split('.')[-1].split(' ')[0]
        # Gets the parameters and writes the string
        current = self._TaxPlot__get_current_parameters()
        params_string = ['%s\t%s' % (att, val) for (att, val) in
                         current.iteritems()]
        params_string[0] = '%s:%s' % (classname, params_string[0])
        return ('\n%s:' % classname).join(sorted(params_string))

    def read_parameters(self, params_str):
        """Reads parameters from a string

        Priority for how competing custom parameters is set is determined by
        priority. When both the current settings and imported settings are not
        default, IMPORTED will use the setting(s) from the file adn CURRENT
        will keep the settings are specified in the class.

        Parameters
        ----------
        params_str : str
            A parsed string in the form of Script:parameter\tvalue that
            specifies parameters for the object.
        """
        # Determines the object class
        class_name = self.__class__
        class_name = '%r' % class_name
        classname = class_name.split('.')[-1].split(' ')[0]
        # Defines the set of classes for which parameters will be considered.
        readable = set([classname, 'TaxPlot'])
        # Determines the parameters which already exist
        current = self._TaxPlot__get_current_parameters()
        default = self._TaxPlot__defaults
        # Reads in the imported parameters
        params = {}
        imported = {}
        lines = [l for l in params_str.split('\n')]
        for l in lines:
            (script, param_set) = l.split(':')
            # Determines the import value
            if script not in readable:
                continue
            (attr, val_str) = param_set.split('\t')
            val = eval(val_str)
            imported[attr] = val
            # Determines the state of the property compared to the default
            imported_default = imported[attr] == default[attr]
            current_default = current[attr] == default[attr]
            if not imported_default and current_default:
                params[attr] = imported[attr]
            elif not (imported_default or current_default):
                if self.priority == 'IMPORTED':
                    params[attr] = imported[attr]
        # Updates the object properties
        self.add_attributes(**params)

    def get_colormap(self):
        """Retuns the colormap and edgecolor as numpy arrays"""
        return self._TaxPlot__colormap, self._TaxPlot__edgecolor

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
        """Makes sure the save location information is sane

        Properties
        -----------
        filename : str
        """
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

    def update_dimensions(self):
        """Updates the axis and figure dimensions for a single axis."""
        self._TaxPlot__fig.set_size_inches(self.fig_dims)
        self._TaxPlot__axes.set_position(self.axis_dims)

    def update_colormap(self):
        """Updates the colormap and egde color"""
        # Check colormap is a sane classs
        no_colors = self.colors is None
        if not no_colors and not isinstance(self.colors, (str, ndarray)):
            raise TypeError('Colormap must a colorbrewer map name or '
                            'numpy array.')

        # Handles the colormap is no data is avaliable
        if self.data is not None:
            if len(self.data.shape) == 1:
                self.data = (array([[1]])*self.data).transpose()
            # Determines the number of rows in the data array
            (num_rows, num_cols) = self.data.shape

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
                elif num_colors >= num_cols and num_cols < num_rows:
                    self._TaxPlot__colormap = self.colors[:(num_cols), :]
                else:
                    raise ValueError('The colormap cannot be used. \nNot'
                                     ' enough colors have been supplied.')

            if self.show_edge:
                self._TaxPlot__edgecolor = zeros(self._TaxPlot__colormap.shape)
            else:
                self._TaxPlot__edgecolor = self._TaxPlot__colormap

    def set_font(self, font_type, font_object):
        """Sets the font object

        Properties
        ----------
        font_type : {tick, title, label, leg, text}
            specifies which font is being update
        font_object : FontProperties
            describes the font being update

        See Also
        --------
        matplotlib FontProperties
        """
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
        self.checkbase()

        # If show_legend is off or there is no potted data, the legend
        # is not updated.
        if self.show_legend and len(self._TaxPlot__patches) > 0:
            # Adds the legend
            patches = reversed(self._TaxPlot__patches)
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
        self.checkbase()

        # A title is not added is there are no axes
        if self._TaxPlot__axes is not None and self.show_title:
            # Checks to see that a title can be added
            title_font = self._TaxPlot__font_set['title']
            self._TaxPlot__title = \
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
            plt.savefig(filename, transparent=True, **self.save_properties)

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
    """Generates a Barchart of the observation and sample data.

    Parameters
    -----------
    data : ndarray
        The data for the object. Each column in data must correspond to a
        sample. Each row must correspond to a value in groups.
    groups : list or 1d ndarray
        The identy of the rows in the array. Typically a taxonomy or other
        observation identity.
    samples : list
        Identifies each column in the array.
    error : ndarray, optional
    filename: String, optional
        Directory where output files should be saved. If filename is not
        specified or set, the figure will not be saved.

    Other Parameters
    ----------------
    bar_width : float (optional)
    x_tick_interval: float (optional)
    y_tick_interval: float
    xlim : float
        default -0.5. The right side of the axis_frame
    x_font_angle: float in [0, 360)
    x_font_align : {'left', 'right', 'center'}
    show_x_labels : bool
    show_y_labels : bool

    See Also
    --------
    TaxPlot, PieChart, MetaTrace, matplotlib.pyplot, CatTableKeys, Colorbrewer

    """
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
                  'xlim': None,
                  'ylim': [0, 1],
                  'xticks': None,
                  'yticks': None,
                  'xticklabels': '',
                  'yticklabels': '',
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
        # Sets up the default values
        if 'fig_dims' not in kwargs:
            kwargs['fig_dims'] = (6, 4)
        if 'axis_dims' not in kwargs:
            kwargs['axis_dims'] = (0.125, 0.1875, 0.75, 0.83334)
        if 'ylim' not in kwargs:
            kwargs['ylim'] = [0, 1]
        # Initialzes the object
        TaxPlot.__init__(self, data, groups, samples, error, filename,
                         **kwargs)
        # Checks the object is sane
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
        self.update_colormap()
        self.update_dimensions()

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

        label_font = self._TaxPlot__font_set['label']
        self._TaxPlot__axes.set_xlabel(self.xlabel, fontproperties=label_font)
        self._TaxPlot__axes.set_ylabel(self.ylabel, fontproperties=label_font)

        # Adds a legend
        self.render_legend()

        # Adds a title
        self.render_title()


class PieChart(TaxPlot):
    """Generates a PieChart of the observation and sample data.

    Parameters
    -----------
    data : ndarray
        The data for the object. Only the first column will be plotted. The
        rows (or values) in data must correspond to the values in groups.
    groups : list or 1d ndarray
        The identy of the rows in the array. Typically a taxonomy or other
        observation identity.
    samples : list
        Identifies each column in the array.
    filename: String, optional
        Directory where output files should be saved. If filename is not
        specified or set, the figure will not be saved.

    Other Parameters
    ----------------
    plot_ccw : bool
        Plot the data counter-clockwise instead of clockwise
    start_angle : float [0, 360)
    show_labels : bool
        Label the pie chart data on the chart
    numeric_labels : bool
        Show the fraction occupied by each wedge as a decimal
    See Also
    --------
    TaxPlot, BarChart, MetaTrace, matplotlib.pyplot, CatTableKeys, Colorbrewer

    """
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
                  'xticklabels': '',
                  'yticklabels': '',
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
        if 'fig_dims' not in kwargs:
            kwargs['fig_dims'] = (5, 3)
        if 'axis_dims' not in kwargs:
            kwargs['axis_dims'] = (0.06, 0.1, 0.48, 0.8)
        if 'show_legend' not in kwargs:
            kwargs['show_legend'] = True
        if 'legend_offset' not in kwargs:
            kwargs['legend_offset'] = (1.8, 0.75)
        if 'xlim' not in kwargs:
            kwargs['xlim'] = [-1.1, 1.1]
        if 'ylim' not in kwargs:
            kwargs['ylim'] = [-1.1, 1.1]
        # Initializes the object
        TaxPlot.__init__(self, data, groups, samples, error=None,
                         filename=filename, **kwargs)
        # Checks the initialized object
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


class MetaTrace(TaxPlot):
    """Generates a trace comparing the distribution of data

    Parameters
    -----------
    data : ndarray
        The data for the object. Rows should correspond to values in groups.
        Columns should be labeled with values in sample.
    groups : list or 1d ndarray
        Continous data corresponding to each row in the array.
    samples : list
        Identifies each column in the data array.
    error : ndarray (optional)
        Error for each point in data
    filename: String, optional
        Directory where output files should be saved. If filename is not
        specified or set, the figure will not be saved.

    Other Parameters
    ----------------
    linestyle : {'None', '-', '--', '-.', ':'}
    marker : {'x', '.', ',', '^', 'v', '>', '<', '1', '2', '3', 's', '4', 's',
              'p', '*', 'h', 'H', '+', 'D', 'd',  '|', '_', 'o', 'None'}

    See Also
    --------
    TaxPlot, BarChart, MetaTrace, matplotlib.pyplot, CatTableKeys, Colorbrewer

    """
    # Sets up properties for the distribtuion
    linestyle = 'None'
    marker = 'x'
    __defaults = {'show_error': True,
                  'fig_dims': (6, 4),
                  'axis_dims': (0.125, 0.1875, 0.5, 0.6),
                  'save_properties': {},
                  'show_edge': False,
                  'colors': None,
                  'show_legend': False,
                  'legend_offset': None,
                  'legend_properties': {},
                  'show_axes': True,
                  'show_frame': True,
                  'xlim': None,
                  'ylim': [0, 1],
                  'xticks': None,
                  'yticks': None,
                  'xticklabels': '',
                  'yticklabels': '',
                  'show_title': False,
                  'title_text': None,
                  'title_properties': {},
                  'use_latex': False,
                  'latex_family': 'sans-serif',
                  'latex_font': ['Helvetica', 'Arial'],
                  'linestyle': 'None',
                  'marker': 'x'}

    def __init__(self, data, groups, samples, error=None, filename=None,
                 **kwargs):
        # Sets up the default values
        if 'fig_dims' not in kwargs:
            kwargs['fig_dims'] = (6, 4)
        if 'axis_dims' not in kwargs:
            kwargs['axis_dims'] = (0.125, 0.1875, 0.75, 0.6)
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
        lineset = set(['None', '-', '--', '-.', ':'])
        markerset = set(['.', ',', '^', 'v', '>', '<', '1', '2', '3', 's',
                         '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd',
                         '|', '_', 'o', 'None'])
        if self.linestyle is not None and self.linestyle not in lineset:
            raise ValueError('%r is not a supported linestyle is not '
                             'supported.' % self.linestyle)
        if self.marker is not None and self.marker not in markerset:
            raise ValueError('%r is not a supported marker style.'
                             % self.marker)

    def render_trace(self):
        """Creates the trace figure"""
        # Checks the input data object is sane
        self.check_trace()
        # Updates variable properties
        self.update_colormap()
        self.update_dimensions()

        # self._TaxPlot__axes.set_visible(True)

        # Sets things up to render using LaTex
        self.render_latex()
        # Plots the data
        for count, category in enumerate(self.samples):
            color = self._TaxPlot__colormap[count, :]
            self._TaxPlot__axes.plot(self.groups, self.data[:, count],
                                     marker='x',
                                     linestyle='None',
                                     color=color)
            self._TaxPlot__axes.errorbar(self.groups, self.data[:, count],
                                         yerr=self.error[:, count],
                                         fmt=None, ecolor=color)

        # Updates the axes with the properties
        tick_font = self._TaxPlot__font_set['tick']
        label_font = self._TaxPlot__font_set['label']
        if self.xlim is not None:
            self._TaxPlot__axes.set_xlim(self.xlim)
        if self.xticks is not None:
            self._TaxPlot__axes.set_xticks(self.xticks)
        if self.xticklabels is not None:
            self._TaxPlot__axes.set_xticklabel(self.xticklabels,
                                               fontproperties=tick_font)
        if self.xlabel is not None:
            self._TaxPlot__axes.set_xlabel(self.xlabel,
                                           fontproperties=label_font)
        if self.ylim is not None:
            self._TaxPlot__axes.set_ylim(self.ylim)
        if self.yticks is not None:
            self._TaxPlot__axes.set_yticks(self.yticks)
        if self.yticklabels is not None:
            self._TaxPlot__axes.set_yticklabel(self.yticklabels,
                                               fontproperties=tick_font)
        if self.ylabel is not None:
            self._TaxPlot__axes.set_ylabel(self.ylabel,
                                           fontproperties=label_font)

        