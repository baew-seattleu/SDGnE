#Author: Sartaj Bhuvaji

import pandas as pd
import pkg_resources

def download_demodata():
    """
    Downloads the demo data from the 'sdgne.demodata' package and returns it as a pandas DataFrame.
    
    Returns:
        pandas.DataFrame: The downloaded demo data.
    """
    data_file_path = pkg_resources.resource_filename('sdgne.demodata', 'data.csv')
    data = pd.read_csv(data_file_path)
    
    return data

    