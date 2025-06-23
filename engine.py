'''
In this file we will define all of the necessary functionality.
'''

# Importing necessary modules
try:
    # Dask
    from dask.distributed import Client, progress
    from dask import delayed
    import dask.array as da
    import dask.dataframe as dd

    # Other
    import pandas as pd

except Exception:
    print('Error: Some or all modules were not properly imported')
    exit()



# -------- K-MEANS FUNCTIONS ----------
def get_first_sample(data_path):
    '''
    Params:
        data_path : str
            Path to a given data file.
    Output:
        Returns the first row of the given data.
    '''
    # Read only first row using pandas
    return pd.read_csv(data_path, nrows=1)



def cost_function(C, X):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the K-means cost function, evaluated
        over the set of points X with respect to the
        centroids C.
    '''
    return 