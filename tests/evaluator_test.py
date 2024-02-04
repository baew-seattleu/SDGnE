#Author: Sartaj Bhuvaji

import unittest
import pandas as pd
import os
import sys

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdgne.datagenerator.autoencoder import BalancedAutoencoder, HeavyDecoderAutoencoder, SingleEncoderAutoencoder
from sdgne.demodata.demodataset import download_demodata
from sdgne.evaluator.evaluator import Evaluation
from sdgne.evaluator.evaluator import GretelEvaluation

class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.data = download_demodata()
        self.minority_column_label = 'class'
        self.minority_class_label = 0
        balanced_autoencoder = BalancedAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        self.new_df = balanced_autoencoder.data_generator()

    def test_duplicate_in_rows(self):
        evaluator = Evaluation(self.new_df, self.minority_column_label, self.minority_class_label)
        self.assertEqual(type(evaluator.duplicate_in_rows()), int) 

    def test_mean_and_std(self):
        evaluator = Evaluation(self.new_df, self.minority_column_label, self.minority_class_label)
        self.assertEqual(type(evaluator.mean_and_std()), pd.DataFrame)

    def test_plot_kde_density_graph(self):
        evaluator = Evaluation(self.new_df, self.minority_column_label, self.minority_class_label)
        values = evaluator.plot_kde_density_graph()
        self.assertEqual(type(values), list)
        self.assertEqual(type(values[1]), pd.DataFrame)
        self.assertEqual(type(values[2]), float)
        self.assertEqual(type(values[3]), float)
        self.assertEqual(type(values[4]), float)

    def test_calculate_metrics(self):
        evaluator = Evaluation(self.new_df, self.minority_column_label, self.minority_class_label)
        self.assertEqual(type(evaluator.calculate_metrics()), dict)

if __name__ == '__main__':
    unittest.main()        