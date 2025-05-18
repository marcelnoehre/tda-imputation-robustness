from sklearn.datasets import fetch_openml

def get_data(id):
    return fetch_openml(data_id=id, as_frame=True)
