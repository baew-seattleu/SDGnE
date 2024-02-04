#Author: Sartaj Bhuvaji

import pkg_resources
import pandas as pd
import keras

class AutoEncoderModel:
    def build_model(self, minority_df: pd.DataFrame, encoder_dense_layers: list, bottle_neck:int,
                        decoder_dense_layers:list, decoder_activation:str) -> keras.Model:
            """
            Build an autoencoder model using the given parameters.

            Parameters:
            minority_df (pd.DataFrame): The input DataFrame for the autoencoder.
            encoder_dense_layers (list): List of integers representing the number of units in each encoder dense layer.
            bottle_neck (int): The number of units in the bottleneck layer.
            decoder_dense_layers (list): List of integers representing the number of units in each decoder dense layer.
            decoder_activation (str): The activation function for the decoder output layer.

            Returns:
            keras.Model: The autoencoder model.
            """
            input_dim = minority_df.shape[1]
            encoder_input = keras.Input(shape=(input_dim,), name="encoder")
            x = keras.layers.Flatten()(encoder_input)

            for units in encoder_dense_layers:
                x = keras.layers.Dense(units, activation="relu")(x)

            encoder_output = keras.layers.Dense(bottle_neck, activation="relu")(x)
            encoder = keras.Model(encoder_input, encoder_output, name="encoder")

            decoder_input = keras.Input(shape=(bottle_neck,), name="decoder")
            x = decoder_input    

            for units in decoder_dense_layers:
                x = keras.layers.Dense(units, activation="relu")(x)

            decoder_output = keras.layers.Dense(input_dim, activation=decoder_activation)(x)
            decoder = keras.Model(decoder_input, decoder_output, name="decoder")

            autoencoder_input = keras.Input(shape=(input_dim,), name="input")
            encoded = encoder(autoencoder_input)
            decoded = decoder(encoded)
            autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

            return autoencoder


class AutoencoderBase:
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str,
                 encoder_dense_layers: list, bottle_neck: int, decoder_dense_layers: list, decoder_activation: str) -> None:
        self.original_df = original_df
        self.minority_column_label = minority_column_label
        self.minority_class_label = minority_class_label
        self.minority_df = None
        self.majority_df = None
        self.encoder_dense_layers = encoder_dense_layers
        self.bottle_neck = bottle_neck
        self.decoder_dense_layers = decoder_dense_layers
        self.decoder_activation = decoder_activation
        if 'synthetic_data' in original_df.columns:
            original_df.drop(columns=['synthetic_data'], inplace=True)


    def pre_processing(self) -> pd.DataFrame:
            """
            Pre-processes the original dataframe by dropping rows with missing values,
            adding a 'synthetic_data' column, separating the minority and majority data,
            and dropping the 'class' column from the minority dataframe.

            Returns:
                pd.DataFrame: The pre-processed dataframe.
            """
            self.original_df = self.original_df.dropna()
            self.original_df['synthetic_data'] = 0
            self.minority_df = self.original_df[self.original_df[self.minority_column_label] == self.minority_class_label].copy()
            self.majority_df = self.original_df[self.original_df[self.minority_column_label] != self.minority_class_label].copy()
            self.minority_df.drop(columns=['class'], inplace=True)
            return self.original_df


    def generate_synthetic_data(self, model_name: str, no_of_syntetic_data:int) -> pd.DataFrame:
            """
            Generates synthetic data using an autoencoder model.

            Args:
                model_name (str): The name of the autoencoder model.
                no_of_syntetic_data (int, optional): The number of synthetic data points to generate. 
                    If not provided or set to 0, the number of synthetic data points will be determined based on the minority and majority data. 
                    Defaults to .minority_df.shape[0] * 2

            Returns:
                pd.DataFrame: The generated synthetic data.
            """
            model_path = pkg_resources.resource_filename('sdgne.models', f'{model_name}.h5')
            autoencoder_model = keras.models.load_model(model_path)
            autoencoder_model.compile(optimizer='adam', loss='mse')

            if no_of_syntetic_data <= 0 and (self.majority_df.shape[0] - self.minority_df.shape[0]) == 0:
                no_of_syntetic_data = self.minority_df.shape[0] * 2

            if no_of_syntetic_data <= 0 and (self.majority_df.shape[0] - self.minority_df.shape[0]) > 0:
                no_of_syntetic_data = self.majority_df.shape[0] - self.minority_df.shape[0]

            generated_data = pd.DataFrame()  
            while generated_data.shape[0] < no_of_syntetic_data:
                new_data = autoencoder_model.predict(self.minority_df, verbose=0)  
                new_data = pd.DataFrame(new_data, columns=self.minority_df.columns)
                generated_data = pd.concat([generated_data, new_data])

            generated_data = generated_data[:no_of_syntetic_data]
            generated_data['class'] = 0
            generated_data['synthetic_data'] = 1
            return generated_data

    def get_model(self) -> keras.Model:
            """
            Returns the autoencoder model.

            Returns:
                keras.Model: The autoencoder model.
            """
            model = AutoEncoderModel()
            autoencoder_model = model.build_model(self.minority_df, self.encoder_dense_layers, self.bottle_neck,
                                                  self.decoder_dense_layers, self.decoder_activation)
            return autoencoder_model

    def data_generator(self, model_name: str, no_of_syntetic_data: int) -> pd.DataFrame:
            """
            Generates synthetic data using the specified model.

            Args:
                model_name (str): The name of the model to use for generating synthetic data.
                no_of_syntetic_data (int, optional): The number of synthetic data points to generate. Defaults to 0.

            Returns:
                pd.DataFrame: The generated synthetic data concatenated with the original data.
            """
            self.pre_processing()
            synthetic_data = self.generate_synthetic_data(model_name, no_of_syntetic_data)
            new_df = pd.concat([self.original_df, synthetic_data], axis=0)
            return new_df


class BalancedAutoencoder(AutoencoderBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'balanced_autoencoder'
        encoder_dense_layers = [22, 20]
        bottle_neck = 16
        decoder_dense_layers = [20, 22]
        decoder_activation = 'sigmoid'
        super().__init__(original_df, minority_column_label, minority_class_label,
                         encoder_dense_layers, bottle_neck, decoder_dense_layers, decoder_activation)

    def data_generator(self, no_of_syntetic_data: int = 0) -> pd.DataFrame:
        return super().data_generator(self.model_name, no_of_syntetic_data)


class HeavyDecoderAutoencoder(AutoencoderBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'heavy_decoder_autoencoder'
        encoder_dense_layers = [22, 20]
        bottle_neck = 16
        decoder_dense_layers = [18, 20, 22, 24]
        decoder_activation = 'sigmoid'
        super().__init__(original_df, minority_column_label, minority_class_label,
                         encoder_dense_layers, bottle_neck, decoder_dense_layers, decoder_activation)

    def data_generator(self, no_of_syntetic_data: int=0) -> pd.DataFrame:
        return super().data_generator(self.model_name, no_of_syntetic_data)


class SingleEncoderAutoencoder(AutoencoderBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'single_encoder_autoencoder'
        encoder_dense_layers = [20]
        bottle_neck = 16
        decoder_dense_layers = [18, 20]
        decoder_activation = 'sigmoid'
       
        super().__init__(original_df, minority_column_label, minority_class_label,
                         encoder_dense_layers, bottle_neck, decoder_dense_layers, decoder_activation)

    def data_generator(self, no_of_syntetic_data: int=0) -> pd.DataFrame:
        return super().data_generator(self.model_name, no_of_syntetic_data)
