#Author: Siddheshwari Bankar

import unittest
import pandas as pd
import os
import sys

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdgne.datagenerator.smote import SMOTE, SDD_SMOTE, Gamma_SMOTE, Gaussian_SMOTE, Gamma_BoostCC,  SDD_BoostCC, ANVO
from sdgne.demodata.demodataset import download_demodata

class smotetest(unittest.TestCase):
    def setUp(self):
        self.data = download_demodata()
        self.minority_column_label = 'class'
        self.minority_class_label = 0

    def test_smote_with_no_of_syntetic_data_100(self):
        smote = SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = smote.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_sdd_smote_with_no_of_syntetic_data_100(self):
        sdd_smote = SDD_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = sdd_smote.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_gamma_smote_with_no_of_syntetic_data_100(self):
        gamma_smote = Gamma_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gamma_smote.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_gaussian_smote_with_no_of_syntetic_data_100(self):
        gaussian_smote = Gaussian_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gaussian_smote.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)
        
    def test_gamma_boostCC_with_no_of_syntetic_data_100(self):
        gamma_boostCC = Gamma_BoostCC(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gamma_boostCC.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)    
    
    def test_sdd_boostCC_with_no_of_syntetic_data_100(self):
        sdd_boostCC = SDD_BoostCC(self.data, self.minority_column_label, self.minority_class_label)
        new_df = sdd_boostCC.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)    
         
    def test_anvo_with_no_of_syntetic_data_100(self):
        anvo = ANVO(self.data, self.minority_column_label, self.minority_class_label)
        new_df = anvo.data_generator(num_to_synthesize=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+1)     

    def test_anvo(self):
        anvo = ANVO(self.data, self.minority_column_label, self.minority_class_label)
        new_df = anvo.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])

    def test_smote(self):
        smote = SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = smote.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])

    def test_sdd_smote(self):
        sdd_smote = SDD_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = sdd_smote.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])

    def test_gamma_smote(self):
        gamma_smote = Gamma_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gamma_smote.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])  
    
    def test_gaussian_smote(self):
        gaussian_smote = Gaussian_SMOTE(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gaussian_smote.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1]) 
    
    def test_gamma_boostCC(self):
        gamma_boostCC = Gamma_BoostCC(self.data, self.minority_column_label, self.minority_class_label)
        new_df = gamma_boostCC.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1]) 
    
    def test_sdd_boostCC(self):
        sdd_boostCC = SDD_BoostCC(self.data, self.minority_column_label, self.minority_class_label)
        new_df = sdd_boostCC.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1]) 
    
if __name__ == '__main__':
    unittest.main()