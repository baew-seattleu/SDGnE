#Author: Sartaj Bhuvaji

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, entropy
from gretel_client.projects.models import read_model_config
from gretel_client.helpers import poll
from gretel_client.config import RunnerMode
from gretel_client.evaluation.quality_report import QualityReport
from gretel_client import configure_session

class Evaluation:
    def __init__(self, dataframe: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.dataframe = dataframe
        self.minority_column_label = minority_column_label
        self.minority_class_label = minority_class_label
        self._pre_processing()

    def _pre_processing(self) -> pd.DataFrame:
        if not 'synthetic_data' in self.dataframe.columns:
            print("Synthetic data column not found in dataframe. Please add column {'synthetic_data' = 1/0} column to dataframe.")
            return 

        self.dataframe.dropna(inplace=True)
        self.original_minority_data = self.dataframe[(self.dataframe[self.minority_column_label] == self.minority_class_label) & (self.dataframe['synthetic_data'] == 0)]
        self.synthetic_minority_data = self.dataframe[(self.dataframe[self.minority_column_label] == self.minority_class_label) & (self.dataframe['synthetic_data'] == 1)]

    def duplicate_in_rows(self) -> float:
        """
        Calculates the percentage of duplicate rows between the original minority data and the synthetic minority data.

        Returns:
            float: The percentage of duplicate rows.
        """
        df1 = self.original_minority_data.copy()
        df2 = self.synthetic_minority_data.copy()
        merged_df = pd.merge(df1, df2)
        total_rows = merged_df.shape[0]
        
        df1 = df1.round(4)
        df2 = df2.round(4)

        if total_rows == 0:
            return 0

        duplicate_count = total_rows - merged_df.drop_duplicates().shape[0]
        duplicate_percentage = (duplicate_count / total_rows) * 100

        return duplicate_percentage
    
    def mean_and_std(self) -> pd.DataFrame:
            """
            Calculate the mean and standard deviation for each column in the original and synthetic minority data.

            Returns:
                pd.DataFrame: A DataFrame containing the mean difference, mean of original minority data,
                              mean of synthetic minority data, standard deviation of original minority data,
                              and standard deviation of synthetic minority data for each column.
            """
            output = {}
            original_minority_data_cp = self.original_minority_data.copy()
            synthetic_minority_data_cp = self.synthetic_minority_data.copy()

            original_minority_data_cp.drop(columns=['synthetic_data', 'class'], inplace=True)
            synthetic_minority_data_cp.drop(columns=['synthetic_data', 'class'], inplace=True)

            for column in original_minority_data_cp.columns:
                mean_df1 = original_minority_data_cp[column].mean()
                std_df1  = original_minority_data_cp[column].std()
                mean_df2 = synthetic_minority_data_cp[column].mean()
                std_df2  = synthetic_minority_data_cp[column].std()
                meandiff = abs(mean_df1 - mean_df2)

                output[column] = {'Mean_diff': meandiff,
                                'Mean_original_minority_data': mean_df1, 'Mean_synthetic_minority_data': mean_df2,
                                'Std_original_minority_data' : std_df1,  'Std_synthetic_minority_data' : std_df2}
                
            return pd.DataFrame(output).transpose().sort_values(by='Mean_diff', ascending=False)


    def plot_kde_density_graph(self) -> list:
        """
        Plots the kernel density estimation (KDE) graphs for each column of the original and synthetic minority data.
        Calculates the highlighted area and KL divergence for each column.
        
        Returns a list containing the following:
        - plt: The matplotlib.pyplot object containing the KDE graphs.
        - column_details: A pandas DataFrame with details of each column, including the highlighted area and KL divergence.
        - total_highlighted_area: The sum of all highlighted areas.
        - total_kl_divergence: The sum of all KL divergences.
        - average_kl_divergence: The average KL divergence across all columns.
        """
        df1 = self.original_minority_data.copy()
        df2 = self.synthetic_minority_data.copy() 

        df1.drop(columns=['synthetic_data', 'class'], inplace=True)
        df2.drop(columns=['synthetic_data', 'class'], inplace=True)
        
        num_columns = len(df1.columns)
        num_rows = int(np.ceil(num_columns / 5))
        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 2 * num_rows))

        highlighted_areas = {} 
        kl_divergences = {}  

        for i, column in enumerate(df1.columns):
            row = i // 5
            col = i % 5
            ax = axes[row, col] if num_rows > 1 else axes[col]

            sns.kdeplot(data=df1[column], color='blue', label='original_df', ax=ax)
            sns.kdeplot(data=df2[column], color='green', label='synthetic_df', ax=ax)

            x = np.linspace(0, 1, 1000)  
            kde_original = gaussian_kde(df1[column])
            kde_synthetic = gaussian_kde(df2[column])
            y1 = kde_original(x)
            y2 = kde_synthetic(x)
            ax.fill_between(x, y1, y2, where=(y1 > y2), interpolate=True, color='lightcoral', alpha=0.3)
            ax.fill_between(x, y1, y2, where=(y1 <= y2), interpolate=True, color='lightgreen', alpha=0.3)

            highlighted_area = np.sum(np.maximum(y1 - y2, 0) * np.diff(x)[0])
            highlighted_areas[column] = highlighted_area

            # Calculate and store the KL divergence for the column
            # REF https://www.kaggle.com/code/nhan1212/some-statistical-distances
            kl_divergence = entropy(y1, y2) 
            kl_divergences[column] = kl_divergence

            ax.set_xlabel(column)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 3.5)
            ax.legend()

        plt.tight_layout()
        total_highlighted_area = np.sum(list(highlighted_areas.values()))
        total_kl_divergence = np.sum(list(kl_divergences.values()))

        column_details = pd.concat(
            [pd.DataFrame({'Column': column, 'Highlighted area': area, 'KL divergence': divergence}, index=[0])
            for column, area, divergence in zip(highlighted_areas.keys(), highlighted_areas.values(), kl_divergences.values())],
            ignore_index=True
        )

        average_kl_divergence = total_kl_divergence / num_columns   
        return [plt, column_details, float(total_highlighted_area), float(total_kl_divergence), float(average_kl_divergence)]

    
    def calculate_metrics(self) -> dict:
            """
            Calculates and returns the metrics for the dataframe.

            Returns:
                dict: A dictionary containing the calculated metrics.
                      The keys are 'duplicate_percentage' and 'mean_and_std'.
                      The values are the corresponding calculated metrics.
            """
            self.columns = self.dataframe.columns
            duplicate_percentage = self.duplicate_in_rows()
            mean_and_std = self.mean_and_std()
            return {'duplicate_percentage': duplicate_percentage, 'mean_and_std': mean_and_std}

    def plot_heat_maps(self, annot=False) -> plt:
        """
        Plots heat maps for the correlation matrices of the original and synthetic minority data.

        Parameters:
            annot (bool): Whether to annotate the heat maps with the correlation values. Default is False.

        Returns:
            plt: The matplotlib.pyplot object containing the heat maps.
        """
        df1 = self.original_minority_data.copy()
        df2 = self.synthetic_minority_data.copy() 

        df1.drop(columns=['synthetic_data', 'class'], inplace=True)
        df2.drop(columns=['synthetic_data', 'class'], inplace=True)
        
        _ , axes = plt.subplots(1, 2, figsize=(10, 10))
        sns.heatmap(df1.corr(), annot=annot, cmap='viridis', ax=axes[0], cbar=False, square=True)
        axes[0].set_title('Original')

        sns.heatmap(df2.corr(), annot=annot, cmap='viridis', ax=axes[1], cbar=False, square=True)
        axes[1].set_title('Synthetic')
        plt.tight_layout()
        return plt

class GretelEvaluation(Evaluation):
    def __init__(self, dataframe: pd.DataFrame, minority_column_label: str, minority_class_label: str, gretel_api_key:str):
        super().__init__(dataframe, minority_column_label, minority_class_label)
        self.gretel_api_key = gretel_api_key
        pd.set_option("max_colwidth", None)
        try:
            configure_session(api_key=self.gretel_api_key, cache="no", validate=True)
        except Exception as e:
            print("Gretel API key not valid. Please provide valid API key.")
        print("Acknowledgement: This is a quality report for the synthetic data generated by Gretel AI. Please refer to https://gretel.ai/ for complete documentation")


    def run_gretel_quality_report(self) -> dict:
            """
            Runs the Gretel Quality Report on the synthetic data.

            Returns:
                dict: A dictionary containing the quality report generated by Gretel AI.
            """
            report = QualityReport(data_source=self.synthetic_minority_data, ref_data=self.original_minority_data)
            try:
                report.run()
            except Exception as e:
                print("Error in running Gretel Quality Report. Please check the dataframes.")
                return
            gretel_report = {'Acknowledgement': 'This is a quality report for the synthetic data generated by Gretel AI. Please refer to https://gretel.ai/ for complete documentation'}
            gretel_report['Gretel_report'] = report.peek()
            return gretel_report
