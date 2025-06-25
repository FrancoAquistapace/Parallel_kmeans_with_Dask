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
    XXT = da.einsum('ij,ij -> i', X_da, X_da)
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
    XC_term = da.einsum('nj, mj -> nm', X_da, C)

    # Calculate CC term
    CC_term = da.einsum('ij,ij -> i', C, C)
    
    return -2*XC_term + CC_term
    

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


def sample_new_centroids(C, X, XXT, phi, l):
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
        phi : float
            Current cost function for the data X
            and the centroids C.
        l : float or int
            Oversampling factor, must be greater 
            than zero.
    Output:
        Returns new centroids from X, sampled with
        probability:
             p_x = l * D^2 / phi
        where D^2 is the squared distance from x to
        C and phi is the current cost function.
    '''
    # Get D^2
    partial_D_sq = partial_squared_dist_matrix(C, X)
    D_sq = da.add(XXT, da.min(partial_D_sq, axis=1))

    # Get sampling probabilities
    p_X = (l * D_sq / phi).compute_chunk_sizes()
    
    # Draw random numbers between 0 and 1
    rand_nums = da.random.uniform(size=p_X.shape)

    # Get new centroid indexes
    C_prime_idx = rand_nums < p_X

    # Gather new centroids from the data
    C_prime = X.loc[C_prime_idx]
    return C_prime

def get_cluster_classification(C, X, XXT):
    '''
    Params:
        C : Dask array or dataframe
            Array containing centroid locations
        X : Dask array or dataframe
            Array containing data points.
    Output:
        Returns the corresponding centroid in C
        for each sample in X.
    '''
    # Get partial squared distances
    partial_D_sq = partial_squared_dist_matrix(C, X)

    # Select the correct centroid from arg min
    C_ids = da.argmin(partial_D_sq, axis=1)
    return C_ids

def get_centroid_weights(labels):
    '''
    Params:
        labels : array
            Set of labels indicating the centroid of
            each data point.
    Output:
        Returns the weight of each centroid, defined 
        as the number of samples in the data closer
        to that centroid than to any other centroid.
    '''
    unique_labels, counts = da.unique(da.array(labels), return_counts=True)
    return unique_labels, counts