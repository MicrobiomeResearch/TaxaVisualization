#!/usr/bin/env python
# CatTableKeys.py

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014, The American Gut Project"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

from numpy import mean, median, std, ndarray, sum as nsum, array, sqrt
from build_cat_table import (identify_groups,
                             build_sub_table,
                             identify_sample_group)


def check_meta(meta, samples, category):
    """Checks the meta data is sane"""
    # Checks the metadata object is a 2D dictionary
    if not isinstance(meta, dict):
        raise TypeError('meta must be a 2D dictionary.')

    # Gets the metadata keys
    ref_dict = meta[meta.keys()[0]]
    if not isinstance(ref_dict, dict):
        raise TypeError('meta must be a 2D dictionary')
    else:
        ref_keys = set(ref_dict.keys())

    # Checks the sample entries
    for samp, data in meta.iteritems():
        # Checks the object is a dictionary
        if not isinstance(data, dict):
            raise TypeError('meta must be a 2D dictionary')
        # Checks the metadata keys are consistent
        if not set(data.keys()) == ref_keys:
            raise ValueError('Meta data fields provided differ between '
                             'samples.')

    # Checks that every sample id is respresented in the metadata
    if samples is not None:
        for id_ in samples:
            if id_ not in meta.keys():
                raise ValueError('Not all samples are represented in the '
                                 'meta data')

    # Checks the category is in the mapping data
    if category is not None and category not in ref_keys:
        raise ValueError('%s is not a valid metadata category.' % category)


def check_data_array(data, row_names, col_names, data_id='data',
                     row_id='rows', col_id='cols'):
    """Checks the supplied data array and array names are sane"""

    # Checks the data types
    if not isinstance(data, (list, ndarray)):
        raise TypeError('%s must be a numpy array or list.' % data_id)

    if isinstance(data, list) and isinstance(data[0], (list, float, int)):
        data = array(data)

    elif isinstance(data, list):
        raise TypeError('%s must be a list of lists or a list of numbers.')

    if not isinstance(row_names, (list, ndarray)):
        raise TypeError('%s must be a list or a numpy vector.' % row_id)

    elif isinstance(row_names, ndarray) and len(row_names.shape) > 1:
        raise ValueError('%s must be a list or numpy vector' % row_id)

    if not isinstance(col_names, list):
        raise TypeError('%s must be a list.' % col_id)

    # Checks the dimensions match
    num_n_rows = len(row_names)
    num_n_cols = len(col_names)
    data_shape = data.shape
    if len(data_shape) == 1:
        if not data_shape[0] == num_n_rows:
            raise ValueError('There must be a label for each row in the data.')

    if len(data_shape) == 2:
        num_d_rows = data_shape[0]
        num_d_cols = data_shape[1]
        if not num_d_rows == num_n_rows:
            raise ValueError('There must be a label for each row in the data'
                             ' matrix.')
        elif not num_d_cols == num_n_cols:
            raise ValueError('There must be a label for each column in the '
                             'data matrix.')
    # Returns a data array
    return data


class CatTable:
    """Combines observation data and metadata to create a summary

    OPTIONAL INPUTS:
        data -- a numpy array of sample data. This can be updated using
                    the set_data function. Data is required to get a table_out.
                    Data cannot be added without samples and taxa.

        samples -- a list of the sample names, corresponding to the
                    columns in data. This can be updated using the set_data
                    function. Samples cannot be added without data and taxa.

        taxa -- a list of category names, corresponding to the rows in data.
                    This can be updated using the set_data function. Taxa
                    cannot be added without data and samples.

        meta -- a two-dimensional dictionary which keys sample ids to their
                    metadata. This can be updated with the set_meta function.
                    It is required for 'GROUP' type data. (See ATTRIBUTES:
                    data_type)

    ATTRIBUTES:
        data_type -- describes what data should be considered
                    'POP'   : look at the entire table
                    'ID'    : look at a single sample (defined in match_id)
                    'GROUP' : look at a set of samples defined by a meta data
                              category (defined in match_id and category)
                    DEFAULT: 'POP'

        data_mode -- describes how the data should be summarized
                    'ALL' : (all values) values for the target set
                    'AVR' : (average) mean abundance for the target set
                    'BIN' : (binary values) present/absence for the target set
                    'BIS' : (binary sum) the number of counts for the target
                            set
                    'MED' : (median) the median abundance for the target set
                    'SUM' : (sum) the total abundance for the target set
                    DEFAULT: 'ALL'

        error_mode -- describes how error should be handled for the target set
                    of data. Error handling cannot be preformed on 'ID' sets.
                    None  : no error data will be returned
                    'STD' : the standard devation of the target set will be
                            returned
                    'STE' : the standard error of the mean will be returned
                            for the target data set
                    DEFAULT: None

        category -- a metadata category over which to summarize the data.
                    Used in 'GROUP' mode.
                    DEFAULT: None

        match_id -- a group name or sample id to be used as a reference. Used
                    in 'ID' and 'GROUP' mode. For 'ID' data, match_id must be a
                    sample id. For 'GROUP' data, match_id can be sample id, in
                    which case the data set is defined for all samples where
                    the metadata in CATEGORY matches the reference sample, or
                    match_id can be a metadata value for CATEGORY.
                    DEFAULT: None

        name_type -- a string describing how sample ids should be cleaned up.
                    'RAW'   : Does not alter the sample ID
                    'SPLIT' : Uses a delimiter and delimiter position to take
                              only a desired segment of the sample ID. The
                              default delimiter is an underscore (_), and the
                              default position to take is the first position.
                              The delimiter and delimiter position can be
                              changed using set_name_delimiters.
                    'SUB'   : Substitues the value of name_val for the sample
                              id.
                    'CLEAN' : Replaces underscores in the sample id with spaces
                              and capitalizes every word.
                    'S&C'   : Splits the name using the delimiter and delimiter
                              position (as with split), replaces any remaining
                              underscores with spaces and capitalizes every
                              word.
                    DEFAULT: 'RAW'
        name_disp -- indicates whether the match_id or metadata category should
                    be used as the sample name.
                    'ID'    : Use the match_id value as a basis
                    'CAT'   : Use the category value as a basis
                    'DESCR' : Use a discription of what was done to the data
                              (i.e. Average). For functions which return the
                              entire table, DESCR returns the sample ids.
                    DEFUALT: 'ID'

        name_val -- a string or list of strings to substitute for the sample
                    name if the name_type is SUB.
                    DEFAULT: None
        """

    # Sets up intial values
    data_type = 'POP'
    data_mode = 'ALL'
    error_mode = None
    category = None
    group = None
    match_id = None
    name_type = 'RAW'
    name_disp = 'ID'
    __name_delim = '_'
    __name_delim_pos = 0
    name_val = None
    __data = None
    __samples = None
    __taxa = None
    __meta = None
    __table_out = None
    __error_out = None
    __samples_out = None
    __names_out = None
    # Sets up function names

    def __init__(self, data=None, samples=None, taxa=None, meta=None,
                 **kwargs):
        """Initializes an instance of the class"""
        # Adds data arguments
        self._CatTable__data = data
        self._CatTable__samples = samples
        self._CatTable__taxa = taxa
        self._CatTable__meta = meta

        # Sets any positional arguments
        self = self.add_attributes(**kwargs)

        # Checks the object is sane
        self = self.check_cat_table()

    def add_attributes(self, **kwargs):
        """Adds keyword arguments"""
        for (k, v) in kwargs.iteritems():
            if k in KEY_PROPERTIES:
                setattr(self, k, v)
            else:
                raise ValueError('%s is not a CatTable property.' % k)
        return self

    def check_cat_table(self):
        """Checks the validity of a cat table object"""

        # Checks the data type
        if self.data_type not in POSSIBLE_TYPES:
            raise ValueError('The data type is not supported.')

        # Checks the data mode
        if self.data_mode not in FUNCTION_LOOKUP:
            raise ValueError('The data mode is not supported.')

        if self.error_mode is not None and self.error_mode not in ERROR_LOOKUP:
            raise ValueError('The error mode is not supported')

        # Checks the category type is supported
        if not isinstance(self.category, str) and self.category is not None:
            raise TypeError('The category must be a string.')

        # Checks the id is supported
        if not isinstance(self.match_id, str) and self.match_id is not None:
            raise TypeError('The sample to match must be a string.')

        # Checks the name input is sane
        if self.name_type not in POSSIBLE_NAMES:
            raise ValueError('The name type is not supported.')

        # Checks the name display is sane
        if self.name_disp not in POSSIBLE_DISPLAYS:
            raise ValueError('The name display is not supported')

        # Checks the data is sane
        all_none = self._CatTable__data is None and \
            self._CatTable__samples is None and self._CatTable__taxa is None

        some_none = self._CatTable__data is None or \
            self._CatTable__samples is None or self._CatTable__taxa is None

        if not all_none and some_none:
                raise ValueError('Values must be supplied for the data matrix,'
                                 ' samples and columns.')
        if not all_none:
            self._CatTable__data = \
                check_data_array(self._CatTable__data, self._CatTable__taxa,
                                 self._CatTable__samples,
                                 data_id='data',
                                 row_id='taxa',
                                 col_id='samples')

        if self._CatTable__meta is not None:
            check_meta(self._CatTable__meta, self._CatTable__samples,
                       self.category)

        # Checks that arguments are sane based on the data type.
        samples = self._CatTable__samples is not None
        if self.data_type == 'ID':
            if self.match_id is None:
                raise ValueError('A sample ID must be specified for ID data.')
            elif samples and self.match_id not in set(self._CatTable__samples):
                raise ValueError('The sample is not in the dataset.')

        if self.data_type == 'GROUP':
            if self.match_id is None:
                raise ValueError('A sample ID must be specified for GROUP '
                                 'data.')
            if self.category is None:
                raise ValueError('A category must be specified for GROUP '
                                 'data.')

            meta = self._CatTable__meta is not None
            samples = self._CatTable__samples is not None

            if meta and samples:
                possible_groups = identify_groups(mapping=self._CatTable__meta,
                                                  category=self.category)
                name_set = set(self._CatTable__samples).\
                    union(set(possible_groups))
                if self.match_id not in name_set:
                    raise ValueError('The sample or group to match cannot be '
                                     'found.')

        return self

    def define_data_object(self):
        """Creates an output data object"""

        # Gets a list of sample ids to plot
        if self.data_type == 'POP':
            self._CatTable__samples_out = self._CatTable__samples
            self.group = 'Population'

        if self.data_type == 'ID':
            self._CatTable__samples_out = [self.match_id]
            self.group = self.match_id

        if self.data_type == 'GROUP':
            # Determines where samples fall in each group
            group_assign = \
                identify_sample_group(sample_ids=self._CatTable__samples,
                                      mapping=self._CatTable__meta,
                                      category=self.category)
            # Determines if we are matching a group or a sample
            if self.match_id in group_assign:
                self.group = self.match_id
            else:
                self.group = self._CatTable__meta[self.match_id][self.category]

            self._CatTable__samples_out = group_assign[self.group]

        # Creates a subtable of those sample ids
        sub_table = build_sub_table(data=self._CatTable__data,
                                    sample_ids=self._CatTable__samples,
                                    target_ids=self._CatTable__samples_out)

        # Summarizes the data using the data mode
        function = FUNCTION_LOOKUP[self.data_mode][0]
        self._CatTable__table_out = function(sub_table)
        if len(self._CatTable__table_out.shape) == 1:
            self._CatTable__table_out = (array([[1]]) *
                                         self._CatTable__table_out).transpose()

        # Handles the error
        if self.error_mode is None or self.data_type is 'ID':
            self._CatTable__error_out = None
        else:
            error_fun = ERROR_LOOKUP[self.error_mode]
            error_vec = error_fun(sub_table)
            self._CatTable__error_out = (array([[1]]) * error_vec).transpose()

        return self

    def define_name_object(self):
        """Gives a cleaned name for samples"""
        # Sets up the output names in ID mode
        if self._CatTable__samples_out is None:
            raise ValueError('Samples out cannot be None.')

        # Defines the name function
        pos = self._CatTable__name_delim_pos
        delim = self._CatTable__name_delim
        naming = {'RAW': lambda x: x,
                  'SPLIT': lambda x: x.split(delim)[pos],
                  'SUB': lambda x: self.name_val,
                  'CLEAN': lambda x: x.replace('_', ' ').title(),
                  'S&C': lambda x: x.split(delim)[pos].title()}

        name_function = naming[self.name_type]

        # For a single ID, the cleaned ID is returned.
        all_and_bin = self.data_mode == 'ALL' or self.data_mode == 'BIN'
        id_and_desc = self.name_disp == 'ID' or self.name_disp == 'DESCR'
        if self.data_type == 'ID':
            fun_in = self._CatTable__samples_out[0]
            self._CatTable__names_out = [name_function(fun_in)]

        # Cleans up the IDS if the name mode is ID and multiple samples
        # have been supplied.
        elif all_and_bin and id_and_desc:
            self._CatTable__names_out = []
            for id_ in self._CatTable__samples_out:
                self._CatTable__names_out.append(name_function(id_))

        # Cleans up the group name if the name mode is ID and a single group is
        # output. For POP data, the group is designated as 'Population'.
        elif self.name_disp == 'ID' and not (self.data_mode == 'ALL' or
                                             self.data_mode == 'BIN'):
            self._CatTable__names_out = [name_function(self.group)]

        # Handles CATEGORICAL data. For population data, the category is
        # 'Population'
        elif self.name_disp == 'CAT' and all_and_bin:
            num_names = len(self._CatTable__names_out)
            fun_in = self.category
            self._CatTable__names_out = [name_function(fun_in)] * num_names
        elif self.name_disp == 'CAT' and self.data_type == 'GROUP':
            self._CatTable__names_out = [name_function(self.category)]
        elif self.name_disp == 'CAT':
            self._CatTable__names_out = [name_function('Population')]

        # Handles description data
        elif self.name_disp == 'DESCR':
            look_up = FUNCTION_LOOKUP[self.data_mode][1]
            self._CatTable__names_out = [name_function(look_up)]

        return self

    def get_table_out(self):
        """Returns the table for plotting

        OUTPUTS:
            table_out -- a numpy array of the summarized data

            names_out -- a list of supplying names corresponding to each
                        column in table out.

            taxa -- a list of the row names in data

            error_out -- a numpy array of the error for the summarized
                        data or None if no error can be supplied."""

        if self._CatTable__data is None:
            raise ValueError('Data must be supplied to get out a table')

        self = self.check_cat_table()
        self = self.define_data_object()
        self = self.define_name_object()

        return (self._CatTable__table_out,
                self._CatTable__names_out,
                self._CatTable__taxa,
                self._CatTable__error_out)

    def set_name_delimiters(self, delimiter, delim_pos):
        """Sets delimiters for splitting names

        INPUTS:
            delimiter -- a string used to split the strings

            delim_pos -- an integer indicating the portion of split string
                        to be taken. A value of 0 indicates the first
                        position.
        """
        self._CatTable__name_delim = delimiter
        self._CatTable__name_delim_pos = delim_pos
        # Checks the object is sane
        self = self.check_cat_table()
        # Returns the object
        return self

    def set_data(self, data, samples, taxa):
        """Adds data to the object"""
        self._CatTable__data = data
        self._CatTable__samples = samples
        self._CatTable__taxa = taxa
        # Checks the object is sane
        self = self.check_cat_table()
        # Returns the object
        return self

    def set_metadata(self, meta):
        """Adds metadata to the object"""
        self._CatTable__meta = meta
        # Checks the object is sane
        self = self.check_cat_table()
        # Returns the object
        return self


def return_original(data):
    """Returns the original data set"""
    return data


def return_binary(data):
    """Returns a binary version of the data set"""
    if not isinstance(data, (int, float, ndarray)):
        raise TypeError('Data must be numeric')
    return data > 0


def return_binary_sum(data):
    """Returns the number of instances of the data set"""
    if not isinstance(data, (int, float, ndarray)):
        raise TypeError('Data must be numeric')
    return nsum(data > 0, 1)


def return_mean(data):
    """Returns the mean of each row in the data set"""
    if not isinstance(data, (int, float, ndarray)):
        raise TypeError('Data must be numeric')
    return mean(data, 1)


def return_median(data):
    """Returns the mean of each row in the data set"""
    if not isinstance(data, (int, float, ndarray)):
        raise TypeError('Data must be numeric')
    return median(data, 1)


def return_stdev(data):
    """Returns the standard deviation of each row in the data set"""
    if not isinstance(data, ndarray):
        raise TypeError('Data must be an array')
    return std(data, 1)


def return_sterr(data):
    """Returns the standard error of the mean for each row in the data set"""
    if not isinstance(data, ndarray):
        raise TypeError('Data must be an array')
    num_ele = data.shape[1]
    return std(data, 1)/sqrt(num_ele)


def return_sum(data):
    """Returns the sum of each row in the data set"""
    if not isinstance(data, (int, float, ndarray)):
        raise TypeError('Data must be numeric')
    return nsum(data, 1)


KEY_PROPERTIES = set(['data_type', 'data_mode', 'category', 'match_id',
                      'name_type', 'name_val', 'name_disp', 'error_mode'])

POSSIBLE_TYPES = set(['ID', 'POP', 'GROUP'])

POSSIBLE_NAMES = set(['RAW', 'SPLIT', 'SUB', 'CLEAN', 'FUN', 'S&C'])

POSSIBLE_DISPLAYS = set(['ID', 'CAT', 'DESCR'])

FUNCTION_LOOKUP = {'ALL': (return_original, lambda x: x),
                   'BIN': (return_binary, lambda x: x),
                   'BIS': (return_binary_sum, lambda x: 'Counts'),
                   'AVR': (return_mean, lambda x: 'Mean'),
                   'MED': (return_median, lambda x: 'Median'),
                   'SUM': (return_sum, lambda x: 'Sum')}

ERROR_LOOKUP = {'STD': return_stdev,
                'STE': return_sterr}
