#!/usr/bin/env python
# test_build_cat_table.py

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2014"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

from unittest import TestCase, main
from numpy import array, mean, median, std, sum as nsum, sqrt, ndarray
from numpy.testing import assert_array_equal
from CatTableKeys import (check_meta, check_data_array, CatTable,
                          return_original, return_binary, return_binary_sum,
                          return_mean, return_median, return_stdev,
                          return_sum, return_sterr)


class TestCatTableKey(TestCase):

    def setUp(self):
        """Sets up variables for testing"""
        # Sets up the values to test
        self.groups = [(u'k__Bacteria', u' p__Firmicutes'),
                       (u'k__Bacteria', u' p__Bacteroidetes'),
                       (u'k__Bacteria', u' p__Proteobacteria'),
                       (u'k__Bacteria', u' p__Actinobacteria'),
                       (u'k__Bacteria', u' p__Verrucomicrobia'),
                       (u'k__Bacteria', u' p__Tenericutes'),
                       (u'k__Bacteria', u' p__Cyanobacteria'),
                       (u'k__Bacteria', u' p__Fusobacteria')]

        self.samples = ['A_Stark', 'N_Romanov', 'Z.Washburne', 'B_Allen',
                        'F_Smythe', 'J_COBB', 'C_Xavier', 'S_Summers']

        self.data = array([[0.4738, 0.5646, 0.6382, 0.6170, 0.5180, 0.5609,
                            0.6557, 0.5105],
                           [0.3755, 0.3670, 0.2232, 0.3114, 0.3991, 0.3434,
                            0.2122, 0.2694],
                           [0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                            0.0675, 0.0872],
                           [0.0511, 0.0448, 0.0239, 0.0000, 0.0000, 0.0365,
                            0.0344, 0.0376],
                           [0.0159, 0.0000, 0.0249, 0.0000, 0.0000, 0.0085,
                            0.0025, 0.0000],
                           [0.0036, 0.0137, 0.0000, 0.0200, 0.0000, 0.0065,
                            0.0041, 0.0072],
                           [0.0000, 0.0000, 0.0089, 0.0081, 0.0008, 0.0036,
                            0.0055, 0.0038],
                           [0.0000, 0.0000, 0.0676, 0.0105, 0.0000, 0.0270,
                            0.0181, 0.0842]])

        self.mapping = {'A_Stark': {'#SAMPLEID': 'A_Stark', 'SEX': 'male',
                                    'VERSE': 'Marvel', 'AGE': '40', 'POWER':
                                    'No', 'HOME': 'New York', 'SERIES':
                                    'Avengers'},
                        'N_Romanov': {'#SAMPLEID': 'N_Romanov', 'SEX':
                                      'female', 'VERSE': 'Marvel', 'AGE': '35',
                                      'POWER': 'Yes', 'HOME': 'New York',
                                      'SERIES': 'Avengers'},
                        'Z.Washburne': {'#SAMPLEID': 'Z.Washburne', 'SEX':
                                        'female', 'VERSE': 'Wheedon', 'AGE':
                                        '38', 'POWER': 'No', 'HOME':
                                        'Serenity', 'SERIES': 'Firefly'},
                        'B_Allen': {'#SAMPLEID': 'B_Allen', 'SEX': 'male',
                                    'VERSE': 'DC', 'AGE': '19', 'POWER': 'NA',
                                    'HOME': 'Central City', 'SERIES': 'Arrow'},
                        'F_Smythe': {'#SAMPLEID': 'F_Smythe', 'SEX': 'female',
                                     'VERSE': 'DC', 'AGE': '22', 'POWER': 'No',
                                     'HOME': 'Starling City', 'SERIES':
                                     'Arrow'},
                        'J_COBB': {'#SAMPLEID': 'J_COBB', 'SEX': 'male',
                                   'VERSE': 'Wheedon', 'AGE': '30', 'POWER':
                                   'No', 'HOME': 'Serenity', 'SERIES':
                                   'Firefly'},
                        'C_Xavier': {'#SAMPLEID': 'C_Xavier', 'SEX': 'male',
                                     'VERSE': 'Marvel', 'AGE': '60', 'POWER':
                                     'Yes', 'HOME': 'Graymalkin Lane',
                                     'SERIES': 'X-Men'},
                        'S_Summers': {'#SAMPLEID': 'S_Summers', 'SEX': 'male',
                                      'VERSE': 'Marvel', 'AGE': '15', 'POWER':
                                      'Yes', 'HOME': 'Graymalkin Lane',
                                      'SERIES': 'X-Men'}}
        self.category = 'SEX'
        self.Case = CatTable(data=self.data,
                             taxa=self.groups,
                             meta=self.mapping,
                             samples=self.samples)

    # Tests check_meta
    def test_check_meta_dict(self):
        """Tests check_meta throws an error when meta is not a dictionary"""
        self.assertRaises(TypeError, check_meta, 'self.mapping',
                          self.samples, self.category)

    def test_check_meta_2D_dict(self):
        """Tests check_meta throws an error when meta is not a 2D-dictionary"""
        self.assertRaises(TypeError, check_meta, self.mapping['A_Stark'],
                          self.samples, self.category)

    def test_check_meta_reference_keys(self):
        """Tests check_meta throws an error when keys are not consistant"""
        # Sets up a problematic test map
        mapping = {'A_Stark': {'#SAMPLEID': 'A_Stark'},
                   'N_Romanov': {'#SAMPLEID': 'N_Romanov', 'SEX': 'female',
                                 'VERSE': 'Marvel', 'AGE': '35', 'POWER':
                                 'Yes', 'HOME': 'New York', 'SERIES':
                                 'Avengers'},
                   'Z.Washburne': {'#SAMPLEID': 'Z.Washburne', 'SEX': 'female',
                                   'VERSE': 'Wheedon', 'AGE': '38', 'POWER':
                                   'No', 'HOME': 'Serenity', 'SERIES':
                                   'Firefly'}}
        self.assertRaises(ValueError, check_meta, mapping, mapping.keys(),
                          self.category)

    def test_check_meta_all_samples_represented(self):
        """"Tests that check_meta throws an error when a sample isnt there"""
        # Sets up test samples
        samples = ['H_Potter']
        self.assertRaises(ValueError, check_meta, self.mapping, samples,
                          self.category)

    def test_check_meta_category_represented(self):
        """Tests that check_meta throws an error with nonsane category"""
        category = 'Sex'
        self.assertRaises(ValueError, check_meta, self.mapping,
                          self.samples, category)

    def test_check_meta_sane_mapping(self):
        """Tests no errors are thrown when the mapping file is sane"""
        check_meta(self.mapping, self.samples, self.category)

    # Tests check_data_array
    def test_check_data_array_data_class(self):
        """Tests check_data_array handles the wrong data class sanely"""
        self.assertRaises(TypeError, check_data_array, 'self.data',
                          self.groups, self.samples)

    def test_check_data_array_nonnumeric_list(self):
        """Tests check_data_array hanldes non-numeric list data sanely"""
        # Uses common cats for data, which is a list of strings
        self.assertRaises(TypeError, check_data_array, self.groups,
                          self.groups, self.samples)

    def test_check_data_array_row_names_class(self):
        """Tests check_data_array handles the wrong row_names class sanely"""
        self.assertRaises(TypeError, check_data_array, self.data,
                          'self.groups', self.samples)

    def test_check_data_array_col_names_class(self):
        """Tests check_data_array handles the wrong col_names class sanely"""
        self.assertRaises(TypeError, check_data_array, self.data,
                          self.groups, 'self.samples')

    def test_check_data_array_vector(self):
        """Tests check_data_array can convert a data vector"""
        data_in = array([1, 2, 3, 4])
        known_data = array([[1], [2], [3], [4]])
        samples = ['Harry']
        groups = ['Snape', 'D_Malfoy', 'Umbridge', 'Voldemort']
        data_out = check_data_array(data_in, groups, samples)
        self.assertTrue(isinstance(data_out, ndarray))
        self.assertEqual(data_out.shape, (4, 1))
        assert_array_equal(known_data, data_out)

    def test_check_data_array_vector_unequal(self):
        """Tests check_data_array handles unequal numpy vectors correctly"""
        self.assertRaises(ValueError, check_data_array, self.data[1, :],
                          self.groups, self.samples)

    def test_check_data_array_matrix_unequal_rows(self):
        """Tests check_data_array handles matrixes with unequals rows sanely"""
        self.assertRaises(ValueError, check_data_array, self.data[:-2, :],
                          self.groups, self.samples)

    def test_data_array_matrix_unequal_cols(self):
        """Tests check_data_array handles matrixes with unequals rows sanely"""
        self.assertRaises(ValueError, check_data_array, self.data[:, :-2],
                          self.groups, self.samples)

    def test_return_original(self):
        """Tests that the original data set is returned"""
        # Sets up the known value
        known = self.data
        # Tests against the funtion
        self.assertTrue((return_original(self.data) == known).all())

    def test_return_binary(self):
        """Tests the binary data set is returned"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_binary(self.mapping)
        # Sets up the known value
        known = self.data > 0
        # Checks numeric data is handled appropriately
        self.assertTrue((return_binary(self.data) == known).all())

    def test_return_binary_sum(self):
        """Tests the binary sum (counts) are returned for numeric data"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_binary_sum(self.mapping)
        # Sets up the known value
        known = nsum(self.data > 0, 1)
        # Checks numeric data is handled appropriately
        self.assertTrue((return_binary_sum(self.data) == known).all())

    def test_return_mean(self):
        """Tests the binary sum (counts) are returned for numeric data"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_mean(self.mapping)
        # Sets up the known value
        known = mean(self.data, 1)
        # Checks numeric data is handled appropriately
        self.assertTrue((return_mean(self.data) == known).all())

    def test_return_median(self):
        """Tests the binary sum (counts) are returned for numeric data"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_median(self.mapping)
        # Sets up the known value
        known = median(self.data, 1)
        # Checks numeric data is handled appropriately
        self.assertTrue((return_median(self.data) == known).all())

    def test_return_stdev(self):
        """Tests the standard deviation is returned sanely for numeric data"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_stdev(self.mapping)
        # Sets up the known value
        known = std(self.data, 1)
        # Checks numeric data is handled appropriately
        self.assertTrue((return_stdev(self.data) == known).all())

    def test_return_sterr(self):
        """Tests the standard error is returned sanely for numeric data"""
        # Checks an error is raised when non-array data is passed.
        with self.assertRaises(TypeError):
            return_sterr(self.mapping)
        known = std(self.data, 1)/sqrt(self.data.shape[1])
        self.assertTrue((return_sterr(self.data) == known).all())

    def test_return_sum(self):
        """Tests the binary sum (counts) are returned for numeric data"""
        # Checks an error is raised when non-numeric data is passed
        with self.assertRaises(TypeError):
            return_sum(self.mapping)
        # Sets up the known value
        known = nsum(self.data, 1)
        # Checks numeric data is handled appropriately
        self.assertTrue((return_sum(self.data) == known).all())

    def test_init(self):
        """Checks the CAT_TABLE initialized properly"""
        # Sets up the known values
        known_data_type = 'POP'
        known_data_mode = 'ALL'
        known_error_mode = None
        known_category = None
        known_match_id = None
        known_name_type = 'RAW'
        known_name_delim = '_'
        known_name_delim_pos = 0
        known_name_val = None
        known_name_disp = 'ID'
        known_data = None
        known_samples = None
        known_taxa = None
        known_meta = None
        known_table_out = None
        known_error_out = None
        known_samples_out = None
        known_names_out = None

        # Initializes a CatTable object
        test_table = CatTable
        # Tests that intial values are correct
        self.assertEqual(test_table.data_type, known_data_type)
        self.assertEqual(test_table.data_mode, known_data_mode)
        self.assertEqual(test_table.error_mode, known_error_mode)
        self.assertEqual(test_table.category, known_category)
        self.assertEqual(test_table.match_id, known_match_id)
        self.assertEqual(test_table.name_type, known_name_type)
        self.assertEqual(test_table._CatTable__name_delim, known_name_delim)
        self.assertEqual(test_table._CatTable__name_delim_pos,
                         known_name_delim_pos)
        self.assertEqual(test_table.name_val, known_name_val)
        self.assertEqual(test_table.name_disp, known_name_disp)
        self.assertEqual(test_table._CatTable__data, known_data)
        self.assertEqual(test_table._CatTable__samples, known_samples)
        self.assertEqual(test_table._CatTable__taxa, known_taxa)
        self.assertEqual(test_table._CatTable__meta, known_meta)
        self.assertEqual(test_table._CatTable__table_out, known_table_out)
        self.assertEqual(test_table._CatTable__error_out, known_error_out)
        self.assertEqual(test_table._CatTable__samples_out, known_samples_out)
        self.assertEqual(test_table._CatTable__names_out, known_names_out)

    def test_add_attributes(self):
        """Tests attributes are correctly added to a CatTable object"""
        # Attempts to initialize an instance of cat table with a keyword
        # argument that cannot be passed
        with self.assertRaises(ValueError):
            test_table = CatTable(school='Hogwarts')

        # Initializes an instance of the CatTable
        test_table = CatTable(data_type='ID',
                              match_id='N_Romanov')
        # Checks the attributes have been added correctly
        self.assertEqual(test_table.data_type, 'ID')
        self.assertEqual(test_table.match_id, 'N_Romanov')

    def test_check_cat_table(self):
        """Tests that check_cat_table calls all the correct arguments"""
        # Checks that an error is called when the data_type is not supported
        with self.assertRaises(ValueError):
            CatTable(data_type='Yer')

        # Checks that an error is called when the data_mode is not supported
        with self.assertRaises(ValueError):
            CatTable(data_mode='a')

        # Checks that an error is called when error_mode is not supported.
        with self.assertRaises(ValueError):
            CatTable(error_mode='Wizard')

        # Checks that an error is called when category is not a string.
        with self.assertRaises(TypeError):
            CatTable(category=['Harry!'])

        # Checks that an error is called when match_id is not a string.
        with self.assertRaises(TypeError):
            CatTable(match_id=['Hermine'])

        # Checks an error is called when name_type is not sane
        with self.assertRaises(ValueError):
            CatTable(name_type='Granger')

        # Checks an error is called when name_disp is not sane
        with self.assertRaises(ValueError):
            CatTable(name_disp='slapped Draco Malfoy')

        # Checks that an error is called when a data-related argument is added
        # without all its componients
        test_table = CatTable()
        test_table._CatTable__data = self.data
        with self.assertRaises(ValueError):
            test_table.check_cat_table()

        test_table._CatTable__data = None
        test_table._CatTable__samples = self.samples
        with self.assertRaises(ValueError):
            test_table.check_cat_table()

        test_table._CatTable__sample = None
        test_table._CatTable__taxa = self.groups
        with self.assertRaises(ValueError):
            test_table.check_cat_table()

        ## Checks that data handling is sane for ID type data
        # Checks that an error is raised when no ID is given to match
        with self.assertRaises(ValueError):
            CatTable(data_type='ID')
        # Checks an error is raised with the supplied ID is not among the
        # sample set provided
        with self.assertRaises(ValueError):
            CatTable(data_type='ID',
                     match_id='P_Coulson',
                     samples=self.samples)

        ## Checks that data handling is sane for GROUP type data
        # Checks an error is raised when no match_id is given
        with self.assertRaises(ValueError):
            CatTable(data_type='GROUP')

        # Checks an error is raised when no category is given
        with self.assertRaises(ValueError):
            CatTable(data_type='GROUP',
                     match_id='P_Coulson')
        # Checks an error is raised when the match_id is not a group name or
        # a sample name.
        with self.assertRaises(ValueError):
            CatTable(data_type='GROUP',
                     match_id='P_Coulson',
                     category='HOME',
                     meta=self.mapping,
                     samples=self.samples,
                     data=self.data,
                     taxa=self.groups)

    def test_define_data_object(self):
        """Checks the output data object is created sanely"""
        ## Tests population mode
        # Defines the known value
        known_samples = self.samples
        known_data = self.data
        # Defines a CatTable instance
        test = CatTable(data_type='POP',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups)
        test = test.define_data_object()
        # Checks the data returned is sane
        self.assertTrue((test._CatTable__table_out == known_data).all())
        self.assertEqual(test._CatTable__samples_out, known_samples)

        ## Tests ID mode
        # Defines the known value
        known_samples = ['A_Stark']
        known_data = array([self.data[:, 0]]).transpose()
        # Defines a CatTable instance
        test = CatTable(data_type='ID',
                        match_id='A_Stark',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups)
        test = test.define_data_object()
        # # Checks the data returned is sane
        self.assertTrue((test._CatTable__table_out == known_data).all())
        self.assertEqual(test._CatTable__samples_out, known_samples)

        ## Tests Group mode
        # Defines the known value
        known_samples = ['A_Stark', 'B_Allen', 'J_COBB', 'C_Xavier',
                         'S_Summers']
        known_data = self.data[:, [0, 3, 5, 6, 7]]
        # Defines a CatTable instance
        test = CatTable(data_type='GROUP',
                        match_id='A_Stark',
                        category='SEX',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test = test.define_data_object()
        # # Checks the data returned is sane
        self.assertTrue((test._CatTable__table_out == known_data).all())
        self.assertEqual(test._CatTable__samples_out, known_samples)

    def test_define_name_object(self):
        """Checks the output name object is created sanely"""
        # Checks that n error is raised when samples out is undefined
        test = CatTable()
        with self.assertRaises(ValueError):
            test.define_name_object()

        ## Tests each of the cleaning functions on a single ID
        # Raw display
        known = ['Z.Washburne']
        test = CatTable(data_type='ID',
                        match_id='Z.Washburne',
                        name_disp='ID',
                        name_type='RAW',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test.define_data_object()
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)
        # Split display
        known = ['Washburne']
        test.name_type = 'SPLIT'
        test._CatTable__name_delim = '.'
        test._CatTable__name_delim_pos = 1
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)
        # Substitution
        known = ['Zoe Washburne nee Allyne']
        test.name_type = 'SUB'
        test.name_val = 'Zoe Washburne nee Allyne'
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)
        # Clean
        known = ['J Cobb']
        test = CatTable(data_type='ID',
                        match_id='J_COBB',
                        name_disp='ID',
                        name_type='CLEAN',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test.define_data_object()
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)
        # Split and Clean
        known = ['Cobb']
        test._CatTable__name_delim = '_'
        test._CatTable__name_delim_pos = 1
        test.name_type = 'S&C'
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)

        ## Checks output for POPULATION mode
        # Tests population display for with all mode
        known = self.samples
        test = CatTable(data_type='POP',
                        data_mode='ALL',
                        name_disp='ID',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test.define_data_object()
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, known)
        # Test population display with a single sample and description mode
        known = self.samples
        test = CatTable(data_type='POP',
                        data_mode='AVR',
                        name_disp='ID',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test.define_data_object()
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, ['Population'])
        # Tests population display for a single sample in Category mode
        known = self.samples
        test = CatTable(data_type='POP',
                        data_mode='AVR',
                        name_disp='CAT',
                        data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        meta=self.mapping)
        test.define_data_object()
        test.define_name_object()
        self.assertEqual(test._CatTable__names_out, ['Population'])

    def test_get_table_out(self):
        """Checks the table for plotting is returned sanely"""
        # Checks an error is raised when no data is supplied
        test = CatTable()
        with self.assertRaises(ValueError):
            test.get_table_out()

        # The data mode as set should return the input object with no error
        (test_table, test_names, test_taxa, test_error) = \
            self.Case.get_table_out()
        self.assertTrue((test_table == self.data).all())
        self.assertEqual(test_names, self.samples)
        self.assertEqual(test_taxa, self.groups)
        self.assertEqual(test_error, None)

        ## If we set the data mode to average and the error to standard
        ## devation, the table should return the average of the table for the
        ## data and the standard devation as the error
        # Sets up knowns
        known_table = (array([[1]]) * return_mean(self.data)).transpose()
        known_error = (array([[1]]) * return_stdev(self.data)).transpose()
        known_names = ['Population']
        # Creates data object
        test = CatTable(data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        data_mode='AVR',
                        error_mode='STD')
        # Runs the function
        (test_table, test_names, test_taxa, test_error) = test.get_table_out()
        # Checks the outputs
        self.assertTrue((test_table == known_table).all())
        self.assertTrue((test_error == known_error).all())
        self.assertEqual(test_names, known_names)
        self.assertEqual(test_taxa, self.groups)

        ## Tests an individual sample
        # Sets up knowns
        known_table = (array([[1]]) * self.data[:, 3]).transpose()
        known_names = ['B_Allen']
        # Creates data object
        test = CatTable(data=self.data,
                        samples=self.samples,
                        taxa=self.groups,
                        data_type='ID',
                        match_id='B_Allen',
                        error_mode='STD')
        # Runs the function
        (test_table, test_names, test_taxa, test_error) = test.get_table_out()
        # Checks the outputs
        self.assertTrue((test_table == known_table).all())
        self.assertEqual(test_error, None)
        self.assertEqual(test_names, known_names)
        self.assertEqual(test_taxa, self.groups)

    def test_set_name_delimiters(self):
        """Checks the name delimiters are set correctly"""
        name_delim = '/'
        delim_pos = 3
        self.Case.set_name_delimiters(name_delim, delim_pos)
        self.assertEqual(self.Case._CatTable__name_delim, name_delim)
        self.assertEqual(self.Case._CatTable__name_delim_pos, delim_pos)

    def test_set_data(self):
        """Checks that data can be added correctly"""
        Case = CatTable()
        Case.set_data(self.data, self.samples, self.groups)
        self.assertTrue((Case._CatTable__data == self.data).all())
        self.assertEqual(Case._CatTable__samples, self.samples)
        self.assertEqual(Case._CatTable__taxa, self.groups)

    def test_set_metadata(self):
        """Checks the metadata is added correctly"""
        Case = CatTable()
        Case.set_metadata(self.mapping)
        self.assertEqual(Case._CatTable__meta, self.mapping)

if __name__ == '__main__':
    main()
