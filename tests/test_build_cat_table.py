#!/usr/bin/env python
# test_build_cat_table.py

__author__ = "Justine Debelius"
__copyright__ = "Copyright 2013, The American Gut Project"
__credits__ = ["Justine Debelius"]
__license__ = "BSD"
__version__ = "unversioned"
__maintainer__ = "Justine Debelius"
__email__ = "Justine.Debelius@colorado.edu"

from unittest import TestCase, main
from numpy import array, flipud
from americangut.build_cat_table import (identify_groups,
                                         get_category_position,
                                         sort_alphabetically,
                                         fuzzy_match,
                                         identify_sample_group,
                                         build_sub_table,
                                         sort_samples,
                                         sort_categories)


class BuildCatTableTest(TestCase):

    def setUp(self):
        """Sets up the data for the tests"""
        # Sets up the values to test
        self.common_cats = [(u'k__Bacteria', u' p__Firmicutes'),
                            (u'k__Bacteria', u' p__Bacteroidetes'),
                            (u'k__Bacteria', u' p__Proteobacteria'),
                            (u'k__Bacteria', u' p__Actinobacteria'),
                            (u'k__Bacteria', u' p__Verrucomicrobia'),
                            (u'k__Bacteria', u' p__Tenericutes'),
                            (u'k__Bacteria', u' p__Cyanobacteria'),
                            (u'k__Bacteria', u' p__Fusobacteria')]

        self.sample_ids = ['A_Stark', 'N_Romanov', 'Z.Washburne', 'B_Allen',
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
                        'J_COBB': {'#SAMPLEID': '00009112', 'SEX': 'male',
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

    def test_identify_groups(self):
        """Tests that groups can be identified sanely"""
        # Sets up the known value
        known_groups = set(['Marvel', 'Wheedon', 'DC'])
        category = 'VERSE'

        # Checks that an error is raised on a problematic input
        with self.assertRaises(ValueError):
            identify_groups(mapping=self.mapping,
                            category='NAME')

        # Identifies the groups from the supplied category
        test_groups = identify_groups(mapping=self.mapping,
                                      category=category)

        self.assertEqual(test_groups, known_groups)

    def test_get_category_position(self):
        """Tests that a category position can be identified sanely"""
        # Sets up test inputs
        test_proteo_cat = 'Proteo'
        test_list_cat = 'A_Stark'
        # Sets up known values
        known_pos_proteo = 2
        known_pos_poll = 0

        # Tests error assertions
        with self.assertRaises(ValueError):
            get_category_position(self.common_cats, 'Bacteria')

        # Calculates the test values
        test_pos_proteo = get_category_position(all_cats=self.common_cats,
                                                cat_descr=test_proteo_cat)

        test_pos_poll = get_category_position(all_cats=self.sample_ids,
                                              cat_descr=test_list_cat)

        # Test the calculated values
        self.assertEqual(test_pos_proteo, known_pos_proteo)
        self.assertEqual(test_pos_poll, known_pos_poll)

    def test_sort_alphabetically(self):
        """Tests that sort_alphabetically is sane"""
        # Defines the known values
        known_alpha = ['A_Stark', 'B_Allen', 'C_Xavier', 'F_Smythe',
                       'J_COBB', 'N_Romanov', 'S_Summers', 'Z.Washburne']
        known_order = [0, 3, 6, 4, 5, 1, 7, 2]

        # Caclulates the test values
        (test_alpha, test_order) = sort_alphabetically(self.sample_ids)

        # Checks the known and test values are equal
        self.assertEqual(known_alpha, test_alpha)
        self.assertEqual(known_order, test_order)

    def test_fuzzy_match(self):
        """Tests that fuzzy_match is sane"""
        # Defines values to test
        target_tuple = 'Fuso'
        target_string = 'Don'
        possible_strings = ['Rose', 'Jack', 'Martha', 'Donna', 'River', 'Amy',
                            'Rory', 'Clara']
        possible_integers = [0, 3, 2, 4]

        # Defines the known value
        known_pos_tup = 7
        known_match_tup = (u'k__Bacteria', u' p__Fusobacteria')
        known_pos_str = 3
        known_match_str = 'Donna'

        # Checks that errors are called appropriately
        with self.assertRaises(ValueError):
            fuzzy_match(target='Wilf',
                        possible_matches=possible_strings)

        with self.assertRaises(ValueError):
            fuzzy_match(target='Bacteria',
                        possible_matches=self.common_cats)

        with self.assertRaises(TypeError):
            fuzzy_match(target=3,
                        possible_matches=possible_integers)

        # Calculates the test values
        [pos_tup, match_tup] = fuzzy_match(target=target_tuple,
                                           possible_matches=self.common_cats)

        [pos_str, match_str] = fuzzy_match(target=target_string,
                                           possible_matches=possible_strings)

        # Checks the values are sane
        self.assertEqual(pos_tup, known_pos_tup)
        self.assertEqual(pos_str, known_pos_str)
        self.assertEqual(match_tup, known_match_tup)
        self.assertEqual(match_str, known_match_str)

    def test_identify_sample_group(self):
        """Tests that samples are sanely assigned to their groups"""
        # Sets up values for testing
        category = 'VERSE'
        wrong_sample_ids = ['00100', 'J_COBB']

        # Sets up the known value
        known_groups = {'Marvel':  ['A_Stark', 'N_Romanov', 'C_Xavier',
                                    'S_Summers'],
                        'DC':      ['B_Allen', 'F_Smythe'],
                        'Wheedon': ['Z.Washburne', 'J_COBB']}

        # Test that errors are appropriately raised
        with self.assertRaises(ValueError):
            identify_sample_group(sample_ids=wrong_sample_ids,
                                  mapping=self.mapping,
                                  category=category)

        # Calculates the test value
        test_groups = identify_sample_group(sample_ids=self.sample_ids,
                                            mapping=self.mapping,
                                            category=category)

        # Verifies the outputs are the sample
        self.assertEqual(test_groups, known_groups)

    def test_build_sub_table(self):
        """Tests that a sub table can be built sanely"""
        # Sets up the values for testing
        test_ids = ['B_Allen', 'F_Smythe']
        # Sets up the known value
        known_data_out = self.data[:, [3, 4]]

        # Runs a sanity check when incorrect values are passed
        with self.assertRaises(ValueError):
            build_sub_table(data=self.data,
                            sample_ids=test_ids,
                            target_ids=test_ids)

        with self.assertRaises(TypeError):
            build_sub_table(data=self.data,
                            sample_ids=test_ids,
                            target_ids=3)

        with self.assertRaises(ValueError):
            build_sub_table(data=self.data,
                            sample_ids=self.sample_ids,
                            target_ids=['Cat', 'Dog'])

        # Calculates the test value
        test_data_out = build_sub_table(data=self.data,
                                        sample_ids=self.sample_ids,
                                        target_ids=test_ids)

        # Checks the output and the known are equal
        self.assertTrue((known_data_out == test_data_out).all())

    def test_sort_samples(self):
        """Tests that samples are sorted sanely"""

        # Checks errors are raised when data is not appropriate passed
        with self.assertRaises(ValueError):
            sort_samples(data=self.data,
                         sample_ids=self.sample_ids[:3])

        with self.assertRaises(ValueError):
            sort_samples(data=self.data,
                         sample_ids=self.sample_ids,
                         sort_key=12)

        # Tests the default method
        known_ids = ['A_Stark', 'B_Allen', 'C_Xavier', 'F_Smythe',
                     'J_COBB', 'N_Romanov', 'S_Summers', 'Z.Washburne']

        known_data = array([[0.4738, 0.6170, 0.6557, 0.5180, 0.5609,
                             0.5646, 0.5105, 0.6382],
                            [0.3755, 0.3114, 0.2122, 0.3991, 0.3434,
                             0.3670, 0.2694, 0.2232],
                            [0.0801, 0.0330, 0.0675, 0.0821, 0.0135,
                             0.0099, 0.0872, 0.0135],
                            [0.0511, 0.0000, 0.0344, 0.0000, 0.0365,
                             0.0448, 0.0376, 0.0239],
                            [0.0159, 0.0000, 0.0025, 0.0000, 0.0085,
                             0.0000, 0.0000, 0.0249],
                            [0.0036, 0.0200, 0.0041, 0.0000, 0.0065,
                             0.0137, 0.0072, 0.0000],
                            [0.0000, 0.0081, 0.0055, 0.0008, 0.0036,
                             0.0000, 0.0038, 0.0089],
                            [0.0000, 0.0105, 0.0181, 0.0000, 0.0270,
                             0.0000, 0.0842, 0.0676]])
        [test_ids, test_data] = sort_samples(self.data, self.sample_ids)
        self.assertEqual(known_ids, test_ids)
        self.assertTrue((known_data == test_data).all())

        # Test sorting in reverse
        known_ids = known_ids[::-1]
        known_data = known_data[:, ::-1]

        [test_ids, test_data] = sort_samples(self.data, self.sample_ids,
                                             reverse=True)

        self.assertEqual(known_ids, test_ids)
        self.assertTrue((known_data == test_data).all())

        # Tests sorting with a key
        sort_key = 2
        known_ids = ['N_Romanov', 'Z.Washburne', 'J_COBB', 'B_Allen',
                     'C_Xavier', 'A_Stark', 'F_Smythe', 'S_Summers']

        known_data = array([[0.5646, 0.6382, 0.5609, 0.6170, 0.6557,
                             0.4738, 0.5180, 0.5105],
                            [0.3670, 0.2232, 0.3434, 0.3114, 0.2122,
                             0.3755, 0.3991, 0.2694],
                            [0.0099, 0.0135, 0.0135, 0.0330, 0.0675,
                             0.0801, 0.0821, 0.0872],
                            [0.0448, 0.0239, 0.0365, 0.0000, 0.0344,
                             0.0511, 0.0000, 0.0376],
                            [0.0000, 0.0249, 0.0085, 0.0000, 0.0025,
                             0.0159, 0.0000, 0.0000],
                            [0.0137, 0.0000, 0.0065, 0.0200, 0.0041,
                             0.0036, 0.0000, 0.0072],
                            [0.0000, 0.0089, 0.0036, 0.0081, 0.0055,
                             0.0000, 0.0008, 0.0038],
                            [0.0000, 0.0676, 0.0270, 0.0105, 0.0181,
                             0.0000, 0.0000, 0.0842]])
        [test_ids, test_data] = sort_samples(self.data, self.sample_ids,
                                             sort_key=sort_key)

        self.assertEqual(known_ids, test_ids)
        self.assertTrue((known_data == test_data).all())

    def test_sort_categories(self):
        # Defines the values to test
        first_cat = 'Proteobacteria'
        sort_method_2 = 'ALPHA'
        sort_method_3 = 'RETAIN'
        sort_method_4 = 'CUSTOM'
        sort_order = ['Proteo', 'Firmicutes', 'detes', 'Tenericutes',
                      'Actino', 'Cyano', 'Prot']

        # Defines the known values
        def_cats = [(u'k__Bacteria', u' p__Proteobacteria'),
                    (u'k__Bacteria', u' p__Tenericutes'),
                    (u'k__Bacteria', u' p__Verrucomicrobia'),
                    (u'k__Bacteria', u' p__Actinobacteria'),
                    (u'k__Bacteria', u' p__Bacteroidetes'),
                    (u'k__Bacteria', u' p__Cyanobacteria'),
                    (u'k__Bacteria', u' p__Firmicutes'),
                    (u'k__Bacteria', u' p__Fusobacteria')]

        m_2_cats = [(u'k__Bacteria', u' p__Proteobacteria'),
                    (u'k__Bacteria', u' p__Actinobacteria'),
                    (u'k__Bacteria', u' p__Bacteroidetes'),
                    (u'k__Bacteria', u' p__Cyanobacteria'),
                    (u'k__Bacteria', u' p__Firmicutes'),
                    (u'k__Bacteria', u' p__Fusobacteria'),
                    (u'k__Bacteria', u' p__Tenericutes'),
                    (u'k__Bacteria', u' p__Verrucomicrobia')]

        m_3_cats = [(u'k__Bacteria', u' p__Proteobacteria'),
                    (u'k__Bacteria', u' p__Firmicutes'),
                    (u'k__Bacteria', u' p__Bacteroidetes'),
                    (u'k__Bacteria', u' p__Actinobacteria'),
                    (u'k__Bacteria', u' p__Verrucomicrobia'),
                    (u'k__Bacteria', u' p__Tenericutes'),
                    (u'k__Bacteria', u' p__Cyanobacteria'),
                    (u'k__Bacteria', u' p__Fusobacteria')]

        m_4_cats = [(u'k__Bacteria', u' p__Proteobacteria'),
                    (u'k__Bacteria', u' p__Firmicutes'),
                    (u'k__Bacteria', u' p__Bacteroidetes'),
                    (u'k__Bacteria', u' p__Tenericutes'),
                    (u'k__Bacteria', u' p__Actinobacteria'),
                    (u'k__Bacteria', u' p__Cyanobacteria'),
                    (u'k__Bacteria', u' p__Proteobacteria')]

        def_data = array([[0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                           0.0675, 0.0872],
                          [0.0036, 0.0137, 0.0000, 0.0200, 0.0000, 0.0065,
                           0.0041, 0.0072],
                          [0.0159, 0.0000, 0.0249, 0.0000, 0.0000, 0.0085,
                           0.0025, 0.0000],
                          [0.0511, 0.0448, 0.0239, 0.0000, 0.0000, 0.0365,
                           0.0344, 0.0376],
                          [0.3755, 0.3670, 0.2232, 0.3114, 0.3991, 0.3434,
                           0.2122, 0.2694],
                          [0.0000, 0.0000, 0.0089, 0.0081, 0.0008, 0.0036,
                           0.0055, 0.0038],
                          [0.4738, 0.5646, 0.6382, 0.6170, 0.5180, 0.5609,
                           0.6557, 0.5105],
                          [0.0000, 0.0000, 0.0676, 0.0105, 0.0000, 0.0270,
                           0.0181, 0.0842]])

        rev_data = flipud(def_data)

        m_2_data = array([[0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                           0.0675, 0.0872],
                          [0.0511, 0.0448, 0.0239, 0.0000, 0.0000, 0.0365,
                           0.0344, 0.0376],
                          [0.3755, 0.3670, 0.2232, 0.3114, 0.3991, 0.3434,
                           0.2122, 0.2694],
                          [0.0000, 0.0000, 0.0089, 0.0081, 0.0008, 0.0036,
                           0.0055, 0.0038],
                          [0.4738, 0.5646, 0.6382, 0.6170, 0.5180, 0.5609,
                           0.6557, 0.5105],
                          [0.0000, 0.0000, 0.0676, 0.0105, 0.0000, 0.0270,
                           0.0181, 0.0842],
                          [0.0036, 0.0137, 0.0000, 0.0200, 0.0000, 0.0065,
                           0.0041, 0.0072],
                          [0.0159, 0.0000, 0.0249, 0.0000, 0.0000, 0.0085,
                           0.0025, 0.0000]])

        m_3_data = array([[0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                           0.0675, 0.0872],
                          [0.4738, 0.5646, 0.6382, 0.6170, 0.5180, 0.5609,
                           0.6557, 0.5105],
                          [0.3755, 0.3670, 0.2232, 0.3114, 0.3991, 0.3434,
                           0.2122, 0.2694],
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

        m_4_data = array([[0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                           0.0675, 0.0872],
                          [0.4738, 0.5646, 0.6382, 0.6170, 0.5180, 0.5609,
                           0.6557, 0.5105],
                          [0.3755, 0.3670, 0.2232, 0.3114, 0.3991, 0.3434,
                           0.2122, 0.2694],
                          [0.0036, 0.0137, 0.0000, 0.0200, 0.0000, 0.0065,
                           0.0041, 0.0072],
                          [0.0511, 0.0448, 0.0239, 0.0000, 0.0000, 0.0365,
                           0.0344, 0.0376],
                          [0.0000, 0.0000, 0.0089, 0.0081, 0.0008, 0.0036,
                           0.0055, 0.0038],
                          [0.0801, 0.0099, 0.0135, 0.0330, 0.0821, 0.0135,
                           0.0675, 0.0872]])

        # Gets the test value
        [test_def_cats, test_def_data] = \
            sort_categories(self.data, self.common_cats, first_cat)

        [test_rev_cats, test_rev_data] = \
            sort_categories(self.data, self.common_cats, first_cat,
                            reverse=True)

        [test_m_2_cats, test_m_2_data] = \
            sort_categories(self.data, self.common_cats, first_cat,
                            sort_method=sort_method_2)

        [test_m_3_cats, test_m_3_data] = \
            sort_categories(self.data, self.common_cats, first_cat,
                            sort_method=sort_method_3)

        [test_m_4_cats, test_m_4_data] = \
            sort_categories(self.data, self.common_cats, first_cat,
                            sort_method=sort_method_4,
                            category_order=sort_order)

        # Checks the categories are coming out correctly ordered
        self.assertEqual(test_def_cats, def_cats)
        self.assertEqual(test_rev_cats, def_cats[::-1])
        self.assertEqual(test_m_2_cats, m_2_cats)
        self.assertEqual(test_m_3_cats, m_3_cats)
        self.assertEqual(test_m_4_cats, m_4_cats)

        self.assertTrue((test_def_data == def_data).all())
        self.assertTrue((test_rev_data == rev_data).all())
        self.assertTrue((test_m_2_data == m_2_data).all())
        self.assertTrue((test_m_3_data == m_3_data).all())
        self.assertTrue((test_m_4_data == m_4_data).all())


if __name__ == '__main__':
    main()
