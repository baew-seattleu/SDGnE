#Author: Sartaj Bhuvaji

import unittest
import os
import sys

# os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sdgne.datagenerator.autoencoder import BalancedAutoencoder, HeavyDecoderAutoencoder, SingleEncoderAutoencoder, AutoEncoderModel, AutoEncoderModel
from sdgne.demodata.demodataset import download_demodata

class AutoEncoderTest(unittest.TestCase):
    def setUp(self):
        self.data = download_demodata()
        self.minority_column_label = 'class'
        self.minority_class_label = 0

    def test_balanced_autoencoder_with_no_of_syntetic_data_100(self):
        balanced_autoencoder = BalancedAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = balanced_autoencoder.data_generator(no_of_syntetic_data=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_heavy_decoder_autoencoder_with_no_of_syntetic_data_100(self):
        heavy_decoder_autoencoder = HeavyDecoderAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = heavy_decoder_autoencoder.data_generator(no_of_syntetic_data=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_single_encoder_autoencoder_with_no_of_syntetic_data_100(self):
        single_encoder_autoencoder = SingleEncoderAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = single_encoder_autoencoder.data_generator(no_of_syntetic_data=100)
        self.assertEqual(len(new_df), len(self.data)+100)
        self.assertEqual(len(new_df.columns), len(self.data.columns)+ 1)

    def test_balanced_autoencoder(self):
        balanced_autoencoder = BalancedAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = balanced_autoencoder.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])

    def test_heavy_decoder_autoencoder(self):
        heavy_decoder_autoencoder = HeavyDecoderAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = heavy_decoder_autoencoder.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])

    def test_single_encoder_autoencoder(self):
        single_encoder_autoencoder = SingleEncoderAutoencoder(self.data, self.minority_column_label, self.minority_class_label)
        new_df = single_encoder_autoencoder.data_generator()
        class_counts = new_df['class'].value_counts()
        self.assertEqual(class_counts[0], class_counts[1])  

    def test_generate_custom_autoencoder(self):
        encoder_dense_layers = [64,32,16]
        bottle_neck = 14
        decoder_dense_layers = [32,64]
        decoder_activation = 'tanh'
        obj = AutoEncoderModel()
        model = obj.build_model(self.data, encoder_dense_layers, bottle_neck, decoder_dense_layers, decoder_activation)
        model.compile(optimizer='adam', loss='mse')
        model.fit(self.data, self.data, epochs=10, batch_size=32, verbose=0)
        self.assertEqual(len(model.layers), 3)

if __name__ == '__main__':
    unittest.main()