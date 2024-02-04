#Author: Sartaj Bhuvaji

import unittest
import pandas as pd
import os
import sys

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sdgne.demodata.demodataset import download_demodata

class DemoDataTest(unittest.TestCase):
    def test_demo_data_loads(self):
        data = download_demodata()
        self.assertEqual(type(data), pd.DataFrame)

    def test_class_column_exists(self):
        data = download_demodata()
        self.assertTrue('class' in data.columns)   

    def test_more_than_one_class(self):
        data = download_demodata()
        self.assertTrue(len(data['class'].unique()) > 1)     

if __name__ == '__main__':
    unittest.main()        