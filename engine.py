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


def get_XXT_term(X):
    '''
    Params:
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the self multiplication term from the
        squared distance matrix formula.
    '''
    # Turn into array
    X_da = da.array(X)
    # Get diagonal of X*X_T
    XXT = da.diag(da.matmul(X_da, da.transpose(X_da)))
    return XXT


def partial_squared_dist_matrix(C, X):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the partial squared distance matrix, 
        evaluated over the set of points X with respect 
        to the centroids C. The partial matrix is defined
        as:
         D'^2 = 2*XC^T + CC^T
    '''
    # Turn into arrays
    X_da = da.array(X)
    
    # Calculate XC term
    XC_term = da.matmul(X_da, da.transpose(C))

    # Calculate CC term
    CC_term = da.einsum('ij,ji -> i', C, da.transpose(C))
    
    return 2*XC_term + CC_term
    

def cost_function(C, X, XXT):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
        XXT : Dask array or dataframe
            Array with same number of rows as X, 
            containing the X*X^T term of the
            squared distance matrix.
    Output:
        Returns the K-means cost function, evaluated
        over the set of points X with respect to the
        centroids C.
    '''
    # First, get partial squared distances
    D2 = partial_squared_dist_matrix(C, X)

    # Minimize over C axis and sum
    D2_min_sum = da.sum(da.min(D2, axis=1))

    # Add to XXT sum and return
    return da.sum(XXT) + D2_min_sum