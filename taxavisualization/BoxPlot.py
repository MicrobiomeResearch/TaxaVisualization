# #!/usr/bin/env python
# BoxPlot.py

from __future__ import division
from os import mkdir, getcwd
from os.path import splitext, split as fsplit, join as pjoin, exists, isfile
from numpy import arange, zeros, ndarray, array, histogram, floor, ceil
from matplotlib import use, rc
use('agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from inspect import getmembers, ismethod
from warnings import warn, filterwarnings
from CatTableKeys import check_data_array
from americangut.make_phyla_plots import translate_colorbrewer


__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014,"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

# Sets up warning activity to always
filterwarnings("always")


class BoxPlot(TaxPlot):
    """DOC STRING HERE"""
    # Sets up custom properties for the axes
    notch = True
    vertical = True
    boxplot_props = {}


#     def __init__(self, data, groups, samples, filename=None, **kwargs):
#         """Initializes a ScatterPlot instance"""
#         # Sets up the figure and axis dimesnions for the boxplot