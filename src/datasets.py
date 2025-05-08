import os
import openml

def init_openml_api():
    """
    Initialize the OpenML API with the API key stored in the environment variable.
    """
    openml.config.apikey = os.getenv('OPENML_API_KEY')

def load_openml_dataset(id):
    """Load a dataset from OpenML by its ID.

    Parameters:
        id (integer): The unique identifier of the dataset on OpenML.

    Returns:
        numpy.ndarray: A NumPy array containing the dataset's features.
    """
    dataset = openml.datasets.get_dataset(id)
    df, *_ = dataset.get_data()
    return df.select_dtypes(include='number').to_numpy()

### Setup ###
init_openml_api()